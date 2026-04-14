"""
2D Cancer Invasion Model - Sundials IDA with Gridap

This implements the coupled PDE system using:
- Gridap's TransientFEOperator with automatic differentiation for Jacobians
- Sundials IDA solver (BDF-based DAE solver) with KLU sparse direct solver
- Adaptive time stepping with sparse Jacobian support
- Multi-field formulation with correct Gridap ODE API

Governing Equations (implicit form for DAE):
- Leader: 0 = α ∂ρ/∂t - ∇·[(δ-γρ)∇ρ - ρ∇f - ηy ρ êy]
- Follower: 0 = a ∂f/∂t - ∇·[(d-gf)∇f - f∇ρ]

Domain: Ω = [0,1] × [0,1]
BCs: Periodic in x, No-flux at y=0, Dirichlet at y=1
"""

using Gridap
using Gridap.FESpaces
using Gridap.ReferenceFEs
using Gridap.Geometry
using Gridap.Fields
using Gridap.CellData
using Gridap.Algebra
using Gridap.ODEs
using Gridap.MultiField
using LinearAlgebra: fillstored!, norm, dot
using SparseArrays
using DifferentialEquations
using Sundials

# Gridap's diffeq_wrappers is not exported by default, so we implement it directly
# Based on Gridap's _DiffEqsWrappers.jl module

"""
    diffeq_wrappers(op::TransientFEOperator, u0)

Return wrapper functions for DifferentialEquations.jl DAEProblem.
Returns: (residual!, jacobian!, mass!, stiffness!)
"""
function diffeq_wrappers(op, u0)
    odeop = get_algebraic_operator(op)
    t0 = 0.0
    us = (u0, zero(u0))  # Tuple with u and du/dt
    odeopcache = allocate_odeopcache(odeop, t0, us)
    
    function _residual!(res, du, u, p, t)
        update_odeopcache!(odeopcache, odeop, t)
        residual!(res, odeop, t, (u, du), odeopcache)
    end
    
    function _jacobian!(jac, du, u, p, gamma, t)
        update_odeopcache!(odeopcache, odeop, t)
        z = zero(eltype(jac))
        fillstored!(jac, z)
        jacobian_add!(jac, odeop, t, (u, du), (1.0, gamma), odeopcache)
    end
    
    function _mass!(mass, du, u, p, t)
        update_odeopcache!(odeopcache, odeop, t)
        z = zero(eltype(mass))
        fillstored!(mass, z)
        jacobian!(mass, odeop, t, (u, du), (0.0, 1.0), odeopcache)
    end
    
    function _stiffness!(stif, du, u, p, t)
        update_odeopcache!(odeopcache, odeop, t)
        z = zero(eltype(stif))
        fillstored!(stif, z)
        jacobian!(stif, odeop, t, (u, du), (1.0, 0.0), odeopcache)
    end
    
    return _residual!, _jacobian!, _mass!, _stiffness!
end

"""
    prototype_jacobian(op::TransientFEOperator, u0)

Allocate a Jacobian matrix prototype with the correct sparsity pattern.
"""
function prototype_jacobian(op, u0)
    odeop = get_algebraic_operator(op)
    t0 = 0.0
    us = (u0, zero(u0))
    odeopcache = allocate_odeopcache(odeop, t0, us)
    return allocate_jacobian(odeop, t0, us, odeopcache)
end

export CancerInvasionIDA, solve_ida, check_solution

# =============================================================================
# Model Parameters
# =============================================================================

Base.@kwdef struct CancerInvasionIDA
    # Time scaling parameters
    α::Float64 = 1.0   # Leader time scale factor
    a::Float64 = 0.44  # Follower time scale factor
    
    # Diffusion parameters
    δ::Float64 = 1.377  # Leader diffusion coefficient
    d::Float64 = 1.377  # Follower diffusion coefficient
    
    # Adhesion parameters
    γ::Float64 = 0.6048  # Leader adhesion coefficient
    g::Float64 = 1.1772  # Follower adhesion coefficient
    
    # Chemotaxis parameters
    ηy::Float64 = 10.0  # Chemotactic factor in y-direction
    
    # Domain
    domain::NTuple{4,Float64} = (0.0, 1.0, 0.0, 1.0)
    partition::NTuple{2,Int} = (50, 50)
    
    # Time span
    tspan::Tuple{Float64,Float64} = (0.0, 1.0)
    
    # Solver tolerances
    reltol::Float64 = 1e-4
    abstol::Float64 = 1e-6
    
    # Time step limits
    dtmin::Float64 = 1e-8
    dtmax::Float64 = 0.1
end

# =============================================================================
# Initial Conditions
# =============================================================================

function initial_condition_leader(x)
    y = x[2]
    return 0.1 * (1.0 - tanh((y - 0.1) / 0.05))
end

function initial_condition_follower(x)
    y = x[2]
    return 0.4 * (1.0 - tanh((y - 0.1) / 0.05))
end

# =============================================================================
# Mesh and Function Spaces Setup
# =============================================================================

function setup_problem(params::CancerInvasionIDA)
    @info "Setting up cancer invasion problem..."
    @info "  Mesh: $(params.partition) elements"
    @info "  Domain: $(params.domain)"
    
    # Create Cartesian mesh
    model = CartesianDiscreteModel(params.domain, params.partition)
    
    # Tag boundaries
    labels = get_face_labeling(model)
    add_tag_from_tags!(labels, "bottom", [1, 2, 5])   # y=0
    add_tag_from_tags!(labels, "top", [3, 4, 6])      # y=1
    add_tag_from_tags!(labels, "left", [7])           # x=0
    add_tag_from_tags!(labels, "right", [8])          # x=1
    
    # Create reference FE (Lagrange P1)
    order = 1
    reffe = ReferenceFE(lagrangian, Float64, order)
    
    # Test space with Dirichlet on top boundary
    V = TestFESpace(
        model,
        reffe,
        conformity=:H1,
        dirichlet_tags=["top"],
        dirichlet_masks=[true]
    )
    
    # Trial space with zero Dirichlet on top
    U = TrialFESpace(V, 0.0)
    
    # Multi-field spaces using TransientMultiFieldFESpace
    Y = MultiFieldFESpace([V, V])  # Test space
    X = TransientMultiFieldFESpace([U, U])  # Trial space (transient version!)
    
    # Triangulation and measure
    Ω = Triangulation(model)
    dΩ = Measure(Ω, 2*order)
    
    # Initial conditions
    ρ0 = interpolate(initial_condition_leader, U)
    f0 = interpolate(initial_condition_follower, U)
    
    # Get free DOF values directly
    ρ0_vals = get_free_dof_values(ρ0)
    f0_vals = get_free_dof_values(f0)
    u0 = vcat(ρ0_vals, f0_vals)
    N = length(ρ0_vals)
    
    # Initial time derivative (zero for consistent initialization)
    du0 = zero(u0)
    
    @info "  DOFs per field: $N"
    @info "  Total DOFs: $(length(u0))"
    
    return model, U, V, X, Y, Ω, dΩ, u0, du0, N
end

# =============================================================================
# Transient FE Operator Definition
# =============================================================================

function create_transient_operator(X, Y, dΩ, params::CancerInvasionIDA)
    @info "Creating transient FE operator with automatic Jacobian..."
    
    # Extract parameters
    α = params.α
    a = params.a
    δ = params.δ
    d_param = params.d
    γ = params.γ
    g = params.g
    ηy = params.ηy
    
    # Define the residual function for the coupled system
    # u is a TransientMultiFieldCellField containing [ρ(t), f(t)]
    # Using the original quasilinear form: res(t, u, du/dt) = 0
    # This is the standard Gridap formulation where res = 0
    function res(t, u, v)
        ρ, f = u  # TransientCellField for each component
        vρ, vf = v  # Test functions
        
        # Time derivatives (Gridap provides ∂t operator)
        dρ_dt = ∂t(ρ)
        df_dt = ∂t(f)
        
        # Leader equation residual:
        # α∂ρ/∂t + (δ-γρ)∇ρ - ρ∇f - ηy ρ êy = 0 (in weak form)
        res_leader = α * dρ_dt * vρ +
                     (δ - γ * ρ) * (∇(vρ) ⋅ ∇(ρ)) -
                     ρ * (∇(vρ) ⋅ ∇(f)) -
                     ηy * ρ * (∇(vρ) ⋅ VectorValue(0.0, 1.0))
        
        # Follower equation residual:
        # a∂f/∂t + (d-gf)∇f - f∇ρ = 0 (in weak form)
        res_follower = a * df_dt * vf +
                       (d_param - g * f) * (∇(vf) ⋅ ∇(f)) -
                       f * (∇(vf) ⋅ ∇(ρ))
        
        return ∫(res_leader + res_follower)dΩ
    end
    
    # Define Jacobians (Gridap uses AD to compute these automatically!)
    function jac(t, u, du, v)
        ρ, f = u
        dρ, df = du  # Direction for Jacobian
        vρ, vf = v
        
        # Jacobian of leader residual w.r.t. [ρ, f]
        # ∂R_ρ/∂ρ: derivative of leader residual in ρ direction
        jac_ρρ = (δ - γ * ρ) * (∇(vρ) ⋅ ∇(dρ)) -
                 γ * dρ * (∇(vρ) ⋅ ∇(ρ)) -
                 dρ * (∇(vρ) ⋅ ∇(f))
        
        # ∂R_ρ/∂f: derivative of leader residual in f direction  
        jac_ρf = -ρ * (∇(vρ) ⋅ ∇(df))
        
        # ∂R_f/∂ρ: derivative of follower residual in ρ direction
        jac_fρ = -f * (∇(vf) ⋅ ∇(dρ))
        
        # ∂R_f/∂f: derivative of follower residual in f direction
        jac_ff = (d_param - g * f) * (∇(vf) ⋅ ∇(df)) -
                 g * df * (∇(vf) ⋅ ∇(f)) -
                 df * (∇(vf) ⋅ ∇(ρ))
        
        return ∫(jac_ρρ + jac_ρf + jac_fρ + jac_ff)dΩ
    end
    
    # Time derivative part of Jacobian (mass matrix)
    function jac_t(t, u, dtu, v)
        dρ_dt, df_dt = dtu
        vρ, vf = v
        
        jac_t_ρ = α * dρ_dt * vρ
        jac_t_f = a * df_dt * vf
        
        return ∫(jac_t_ρ + jac_t_f)dΩ
    end
    
    # Create transient FE operator - Gridap auto-computes Jacobians!
    tfeop = TransientFEOperator(res, (jac, jac_t), X, Y)
    
    @info "  TransientFEOperator created"
    
    return tfeop
end

# =============================================================================
# Gridap to DifferentialEquations.jl DAE Wrapper
# =============================================================================

"""
    create_dae_wrapper(tfeop, u0, du0)

Create wrapper functions for DAEProblem using our diffeq_wrappers implementation.
Uses the implicit form: res(t, u, du/dt) = 0
"""
function create_dae_wrapper(tfeop, u0, du0)
    @info "Creating Gridap → DAEProblem wrapper..."
    
    # Use our diffeq_wrappers implementation
    # These return functions with correct DAEProblem signatures:
    #   res!(res, du, u, p, t)       - residual
    #   jac!(jac, du, u, p, gamma, t) - combined Jacobian: ∂res/∂u + gamma*∂res/∂(du/dt)
    res!, jac!, mass!, stif! = diffeq_wrappers(tfeop, u0)
    
    # Get Jacobian prototype for sparse pattern
    J_proto = prototype_jacobian(tfeop, u0)
    
    @info "  Sparse Jacobian prototype: $(size(J_proto)) with $(nnz(J_proto)) nonzeros"
    @info "  Sparsity: $(round(100*nnz(J_proto)/prod(size(J_proto)), digits=2))%"
    
    return res!, jac!, J_proto
end

# =============================================================================
# Main Solver with IDA
# =============================================================================

function solve_ida(params::CancerInvasionIDA; verbose=true)
    @info "="^70
    @info "Cancer Invasion Solver - Sundials IDA"
    @info "  Gridap native Jacobians + IDA with KLU sparse solver"
    @info "="^70
    
    # Setup problem
    model, U, V, X, Y, Ω, dΩ, u0, du0, N = setup_problem(params)
    
    # Create transient operator
    tfeop = create_transient_operator(X, Y, dΩ, params)
    
    # Create DAE wrapper functions
    res_func!, jac_func!, J_proto = create_dae_wrapper(tfeop, u0, du0)
    
    # Create DAEFunction with sparse Jacobian prototype
    # Note: IDA uses implicit form res(t, u, du/dt) = 0
    dae_func = DAEFunction(res_func!;
        jac=jac_func!,
        jac_prototype=J_proto
    )
    
    # All variables are differential (not algebraic)
    differential_vars = fill(true, length(u0))
    
    # Create DAE problem
    # Format: DAEProblem(f, du0, u0, tspan, p; differential_vars)
    prob = DAEProblem(dae_func, du0, u0, params.tspan, nothing; 
        differential_vars=differential_vars)
    
    @info ""
    @info "Solving with IDA (BDF-based DAE solver)..."
    @info "  Time span: $(params.tspan)"
    @info "  Relative tolerance: $(params.reltol)"
    @info "  Absolute tolerance: $(params.abstol)"
    @info "  Linear solver: KLU (sparse direct)"
    @info "  Max BDF order: 5"
    
    # Solve with IDA using KLU sparse solver
    # KLU is optimal for sparse Jacobian systems
    alg = IDA(linear_solver=:KLU)
    
    sol = DifferentialEquations.solve(prob, alg;
        reltol=params.reltol,
        abstol=params.abstol,
        dtmin=params.dtmin,
        dtmax=params.dtmax,
        progress=verbose,
        progress_steps=10
    )
    
    @info ""
    @info "Solution complete!"
    @info "  Total steps: $(length(sol.t))"
    @info "  Final time: $(sol.t[end])"
    @info "  Status: $(sol.retcode)"
    
    return sol, model, Ω, dΩ, U, N
end

# =============================================================================
# Post-processing and Diagnostics
# =============================================================================

function check_solution(sol, model, Ω, dΩ, U, N; verbose=true)
    @info ""
    @info "Running diagnostics..."
    
    # Compute diagnostics over time
    times = sol.t
    max_rho = Float64[]
    max_f = Float64[]
    total_mass_rho = Float64[]
    total_mass_f = Float64[]
    
    for u in sol.u
        ρ_vals = u[1:N]
        f_vals = u[N+1:end]
        
        push!(max_rho, maximum(ρ_vals))
        push!(max_f, maximum(f_vals))
        
        # Create CellFields for integration
        ρh = FEFunction(U, ρ_vals)
        fh = FEFunction(U, f_vals)
        
        mass_ρ = sum(∫(ρh)dΩ)
        mass_f = sum(∫(fh)dΩ)
        
        push!(total_mass_rho, mass_ρ)
        push!(total_mass_f, mass_f)
    end
    
    if verbose
        @info "Initial state:"
        @info "  max(ρ) = $(max_rho[1])"
        @info "  max(f) = $(max_f[1])"
        @info "  mass(ρ) = $(total_mass_rho[1])"
        @info "  mass(f) = $(total_mass_f[1])"
        
        @info "Final state:"
        @info "  max(ρ) = $(max_rho[end])"
        @info "  max(f) = $(max_f[end])"
        @info "  mass(ρ) = $(total_mass_rho[end])"
        @info "  mass(f) = $(total_mass_f[end])"
        
        @info "Mass change:"
        @info "  Δmass(ρ) = $((total_mass_rho[end] - total_mass_rho[1])/total_mass_rho[1] * 100)%"
        @info "  Δmass(f) = $((total_mass_f[end] - total_mass_f[1])/total_mass_f[1] * 100)%"
    end
    
    return (
        times=times,
        max_rho=max_rho,
        max_f=max_f,
        mass_rho=total_mass_rho,
        mass_f=total_mass_f
    )
end

# =============================================================================
# Main Execution
# =============================================================================

if abspath(PROGRAM_FILE) == @__FILE__
    @info "Running Cancer Invasion Test with Sundials IDA"
    
    # Create model parameters
    params = CancerInvasionIDA(
        α=1.0, a=0.44,
        δ=1.377, d=1.377,
        γ=0.6048, g=1.1772,
        ηy=10.0,
        domain=(0.0, 1.0, 0.0, 1.0),
        partition=(20, 20),  # Smaller for faster testing
        tspan=(0.0, 0.1),    # Short time for testing
        reltol=1e-3,
        abstol=1e-5,
        dtmax=0.01
    )
    
    # Solve
    sol, model, Ω, dΩ, U, N = solve_ida(params)
    
    # Check results
    diagnostics = check_solution(sol, model, Ω, dΩ, U, N)
    
    @info ""
    @info "Test complete!"
end
