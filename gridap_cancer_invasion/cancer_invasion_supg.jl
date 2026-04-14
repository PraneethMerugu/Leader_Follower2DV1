"""
2D Cancer Invasion Model - Sundials IDA with Gridap + SUPG Stabilization

This implements the coupled PDE system with Streamline Upwind Petrov-Galerkin (SUPG) 
stabilization using:
- Gridap's TransientFEOperator with automatic differentiation for Jacobians
- Sundials IDA solver (BDF-based DAE solver) with KLU sparse direct solver
- Spatial-only SUPG for advection-dominated chemotaxis terms
- Manual divergence expansion (Gridap's DIV doesn't support composite expressions)

Governing Equations (implicit form for DAE):
- Leader: 0 = α ∂ρ/∂t - ∇·[(δ-γρ)∇ρ - ρ∇f - ηy ρ êy]
- Follower: 0 = a ∂f/∂t - ∇·[(d-gf)∇f - f∇ρ]

SUPG Stabilization:
- Adds τ*(v·∇test)*R to weak form where:
  - τ = C*h/|v| is stabilization parameter
  - v is chemotactic velocity
  - R is strong residual (spatial-only)

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

# =============================================================================
# Gridap DifferentialEquations Wrappers
# =============================================================================

"""
    diffeq_wrappers(op::TransientFEOperator, u0)

Return wrapper functions for DifferentialEquations.jl DAEProblem.
Returns: (residual!, jacobian!, mass!, stiffness!)
"""
function diffeq_wrappers(op, u0)
    odeop = get_algebraic_operator(op)
    t0 = 0.0
    us = (u0, zero(u0))
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

export CancerInvasionSUPG, solve_supg, check_solution, get_supg_diagnostics

# =============================================================================
# Model Parameters with SUPG
# =============================================================================

Base.@kwdef struct CancerInvasionSUPG
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
    
    # SUPG parameters
    C::Float64 = 0.5    # SUPG stabilization constant (0=C=1 typical)
    h::Float64 = 0.0    # Element size (0=auto-compute from mesh)
    v_epsilon::Float64 = 1e-10  # Velocity magnitude safeguard
    
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

function setup_problem(params::CancerInvasionSUPG)
    @info "Setting up cancer invasion problem with SUPG..."
    @info "  Mesh: $(params.partition) elements"
    @info "  Domain: $(params.domain)"
    @info "  SUPG C: $(params.C)"
    
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
    Y = MultiFieldFESpace([V, V])
    X = TransientMultiFieldFESpace([U, U])
    
    # Triangulation and measure
    Ω = Triangulation(model)
    dΩ = Measure(Ω, 2*order)
    
    # Compute element size (for SUPG parameter)
    if params.h == 0.0
        # Auto-compute from mesh
        h_val = compute_element_size(model, params.domain, params.partition)
        @info "  Auto-computed element size: h = $h_val"
    else
        h_val = params.h
        @info "  User-specified element size: h = $h_val"
    end
    
    # Initial conditions
    ρ0 = interpolate(initial_condition_leader, U)
    f0 = interpolate(initial_condition_follower, U)
    
    # Get free DOF values
    ρ0_vals = get_free_dof_values(ρ0)
    f0_vals = get_free_dof_values(f0)
    u0 = vcat(ρ0_vals, f0_vals)
    N = length(ρ0_vals)
    
    # Initial time derivative (zero for consistent initialization)
    du0 = zero(u0)
    
    @info "  DOFs per field: $N"
    @info "  Total DOFs: $(length(u0))"
    
    return model, U, V, X, Y, Ω, dΩ, u0, du0, N, h_val
end

"""
    compute_element_size(model, domain, partition)

Compute characteristic element size from mesh.
For Cartesian grid: h = sqrt(dx*dy) or min(dx, dy)
"""
function compute_element_size(model, domain, partition)
    nx, ny = partition
    x0, x1, y0, y1 = domain
    dx = (x1 - x0) / nx
    dy = (y1 - y0) / ny
    # Use minimum edge length as element size
    return min(dx, dy)
end

# =============================================================================
# Transient FE Operator with SUPG
# =============================================================================

function create_transient_operator(X, Y, dΩ, params::CancerInvasionSUPG, h_val::Float64)
    @info "Creating transient FE operator with SUPG stabilization..."
    
    # Extract parameters
    α = params.α
    a = params.a
    δ = params.δ
    d_param = params.d
    γ = params.γ
    g = params.g
    ηy = params.ηy
    C = params.C
    v_epsilon = params.v_epsilon
    
    # Pre-compute constant τ scaling factor
    τ_scale = C * h_val
    
    @info "  SUPG parameters: C=$C, h=$h_val, τ_scale=$τ_scale"
    @info "  Velocity safeguard: ε=$v_epsilon"
    
    # Define the residual function with SUPG stabilization
    function res(t, u, v)
        ρ, f = u
        vρ, vf = v
        
        # Time derivatives
        dρ_dt = ∂t(ρ)
        df_dt = ∂t(f)
        
        # ============================================================
        # STANDARD GALERKIN TERMS (unchanged from original)
        # ============================================================
        res_leader_galerkin = α * dρ_dt * vρ +
            (δ - γ * ρ) * (∇(vρ) ⋅ ∇(ρ)) -
            ρ * (∇(vρ) ⋅ ∇(f)) -
            ηy * ρ * (∇(vρ) ⋅ VectorValue(0.0, 1.0))
        
        res_follower_galerkin = a * df_dt * vf +
            (d_param - g * f) * (∇(vf) ⋅ ∇(f)) -
            f * (∇(vf) ⋅ ∇(ρ))
        
        # ============================================================
        # SUPG STABILIZATION (spatial-only strong residuals)
        # ============================================================
        
        # --- Chemotactic Velocity Fields ---
        # Leader velocity: follows follower gradient + directed motion
        ∇f = ∇(f)
        ∇ρ = ∇(ρ)
        
        # Use Operation() to access gradient components (CellField doesn't support getindex)
        v_ρ = Operation(grad -> VectorValue(-grad[1], -grad[2] - ηy))(∇f)
        v_f = Operation(grad -> VectorValue(-grad[1], -grad[2]))(∇ρ)
        
        # --- Velocity Magnitudes ---
        v_ρ_mag_sq = v_ρ ⋅ v_ρ
        v_f_mag_sq = v_f ⋅ v_f
        
        # Safeguarded velocity magnitudes (use Operation for sqrt)
        v_ρ_mag = Operation(magsq -> sqrt(magsq + v_epsilon^2))(v_ρ_mag_sq)
        v_f_mag = Operation(magsq -> sqrt(magsq + v_epsilon^2))(v_f_mag_sq)
        
        # --- Stabilization Parameters τ ---
        τ_ρ = Operation(vmag -> τ_scale / vmag)(v_ρ_mag)
        τ_f = Operation(vmag -> τ_scale / vmag)(v_f_mag)
        
        # --- Strong Residuals (spatial-only via manual divergence) ---
        # R_ρ = -∇·flux_ρ where flux_ρ = (δ-γρ)∇ρ - ρ∇f
        # Using product rule: ∇·[(δ-γρ)∇ρ] = -γ(∇ρ·∇ρ) + (δ-γρ)laplacian(ρ)
        #                           ∇·[ρ∇f] = (∇ρ·∇f) + ρ*laplacian(f)
        # So: ∇·flux_ρ = -γ*(∇ρ·∇ρ) + (δ-γ*ρ)*laplacian(ρ) - (∇ρ·∇f) - ρ*laplacian(f)
        
        # Leader flux divergence (with minus sign for residual)
        lap_ρ = laplacian(ρ)
        lap_f = laplacian(f)
        grad_ρ_sq = ∇ρ ⋅ ∇ρ
        grad_f_sq = ∇f ⋅ ∇f
        grad_ρ_dot_grad_f = ∇ρ ⋅ ∇f
        
        # R_ρ = -DIV(flux_ρ) = spatial strong residual
        # DIV(flux_ρ) = -γ*(∇ρ·∇ρ) + (δ-γρ)*laplacian(ρ) - (∇ρ·∇f) - ρ*laplacian(f)
        div_flux_ρ = -γ * grad_ρ_sq + (δ - γ * ρ) * lap_ρ - grad_ρ_dot_grad_f - ρ * lap_f
        R_ρ = -div_flux_ρ  # Strong residual for leader
        
        # R_f = -DIV(flux_f) where flux_f = (d-gf)∇f - f∇ρ
        # DIV(flux_f) = -g*(∇f·∇f) + (d-gf)*laplacian(f) - (∇f·∇ρ) - f*laplacian(ρ)
        div_flux_f = -g * grad_f_sq + (d_param - g * f) * lap_f - grad_ρ_dot_grad_f - f * lap_ρ
        R_f = -div_flux_f  # Strong residual for follower
        
        # --- SUPG Test Function Gradients ---
        # v · ∇(test)
        ∇vρ = ∇(vρ)
        ∇vf = ∇(vf)
        
        # Use Operation() for component-wise dot products
        v_dot_∇vρ = Operation((v, g) -> v[1]*g[1] + v[2]*g[2])(v_ρ, ∇vρ)
        v_dot_∇vf = Operation((v, g) -> v[1]*g[1] + v[2]*g[2])(v_f, ∇vf)
        
        # --- SUPG Stabilization Terms ---
        # SUPG_ρ = τ_ρ * (v_ρ · ∇vρ) * R_ρ
        # SUPG_f = τ_f * (v_f · ∇vf) * R_f
        supg_leader = τ_ρ * v_dot_∇vρ * R_ρ
        supg_follower = τ_f * v_dot_∇vf * R_f
        
        # --- Total Residual ---
        res_leader_total = res_leader_galerkin + supg_leader
        res_follower_total = res_follower_galerkin + supg_follower
        
        return ∫(res_leader_total + res_follower_total)dΩ
    end
    
    # Define Jacobians (Gridap uses AD to compute these automatically!)
    # We need to provide the structure but AD fills in the values
    function jac(t, u, du, v)
        ρ, f = u
        dρ, df = du
        vρ, vf = v
        
        # Jacobian of leader residual w.r.t. [ρ, f]
        jac_ρρ = (δ - γ * ρ) * (∇(vρ) ⋅ ∇(dρ)) -
            γ * dρ * (∇(vρ) ⋅ ∇(ρ)) -
            dρ * (∇(vρ) ⋅ ∇(f))
        
        jac_ρf = -ρ * (∇(vρ) ⋅ ∇(df))
        
        jac_fρ = -f * (∇(vf) ⋅ ∇(dρ))
        
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
    
    # Create transient FE operator
    tfeop = TransientFEOperator(res, (jac, jac_t), X, Y)
    
    @info "  TransientFEOperator with SUPG created"
    
    return tfeop
end

# =============================================================================
# Gridap to DifferentialEquations.jl DAE Wrapper
# =============================================================================

"""
    create_dae_wrapper(tfeop, u0, du0)

Create wrapper functions for DAEProblem using our diffeq_wrappers implementation.
"""
function create_dae_wrapper(tfeop, u0, du0)
    @info "Creating Gridap → DAEProblem wrapper..."
    
    res!, jac!, mass!, stif! = diffeq_wrappers(tfeop, u0)
    
    # Get Jacobian prototype for sparse pattern
    J_proto = prototype_jacobian(tfeop, u0)
    
    @info "  Sparse Jacobian prototype: $(size(J_proto)) with $(nnz(J_proto)) nonzeros"
    @info "  Sparsity: $(round(100*nnz(J_proto)/prod(size(J_proto)), digits=2))%"
    
    return res!, jac!, J_proto
end

# =============================================================================
# Main Solver with SUPG
# =============================================================================

function solve_supg(params::CancerInvasionSUPG; verbose=true)
    @info "="^70
    @info "Cancer Invasion Solver - Sundials IDA with SUPG"
    @info "  Gridap native Jacobians + IDA with KLU sparse solver"
    @info "  Spatial-only SUPG stabilization"
    @info "="^70
    
    # Setup problem
    model, U, V, X, Y, Ω, dΩ, u0, du0, N, h_val = setup_problem(params)
    
    # Create transient operator with SUPG
    tfeop = create_transient_operator(X, Y, dΩ, params, h_val)
    
    # Create DAE wrapper functions
    res_func!, jac_func!, J_proto = create_dae_wrapper(tfeop, u0, du0)
    
    # Create DAEFunction with sparse Jacobian prototype
    dae_func = DAEFunction(res_func!;
        jac=jac_func!,
        jac_prototype=J_proto
    )
    
    # All variables are differential (not algebraic)
    differential_vars = fill(true, length(u0))
    
    # Create DAE problem
    prob = DAEProblem(dae_func, du0, u0, params.tspan, nothing;
        differential_vars=differential_vars)
    
    @info ""
    @info "Solving with IDA (BDF-based DAE solver)..."
    @info "  Time span: $(params.tspan)"
    @info "  Relative tolerance: $(params.reltol)"
    @info "  Absolute tolerance: $(params.abstol)"
    @info "  Linear solver: KLU (sparse direct)"
    @info "  Max BDF order: 5"
    @info "  SUPG C: $(params.C)"
    
    # Solve with IDA using KLU sparse solver
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
    
    return sol, model, Ω, dΩ, U, N, params.C
end

# =============================================================================
# Post-processing and Diagnostics
# =============================================================================

function check_solution(sol, model, Ω, dΩ, U, N, C_supg; verbose=true)
    @info ""
    @info "Running diagnostics..."
    @info "  SUPG C = $C_supg"
    
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

"""
    get_supg_diagnostics(sol, model, Ω, dΩ, U, N, params)

Compute SUPG-specific diagnostics like velocity fields and stabilization parameters.
"""
function get_supg_diagnostics(sol, model, Ω, dΩ, U, N, params::CancerInvasionSUPG)
    @info ""
    @info "Computing SUPG diagnostics..."
    
    # Get final solution
    u_final = sol.u[end]
    ρ_vals = u_final[1:N]
    f_vals = u_final[N+1:end]
    
    ρh = FEFunction(U, ρ_vals)
    fh = FEFunction(U, f_vals)
    
    # Compute velocity fields
    ∇ρ = ∇(ρh)
    ∇f = ∇(fh)
    
    # Leader velocity: v_ρ = -∇f - (0, ηy)
    v_ρ = VectorValue(-∇f[1], -∇f[2] - params.ηy)
    
    # Follower velocity: v_f = -∇ρ
    v_f = VectorValue(-∇ρ[1], -∇ρ[2])
    
    # Compute velocity magnitudes
    v_ρ_mag = sqrt(v_ρ ⋅ v_ρ + params.v_epsilon^2)
    v_f_mag = sqrt(v_f ⋅ v_f + params.v_epsilon^2)
    
    # Stabilization parameters
    h_val = params.h > 0 ? params.h : compute_element_size(model, params.domain, params.partition)
    τ_ρ = params.C * h_val / v_ρ_mag
    τ_f = params.C * h_val / v_f_mag
    
    # Average values
    avg_v_ρ = sum(∫(v_ρ_mag)dΩ) / sum(∫(1.0)dΩ)
    avg_v_f = sum(∫(v_f_mag)dΩ) / sum(∫(1.0)dΩ)
    avg_τ_ρ = sum(∫(τ_ρ)dΩ) / sum(∫(1.0)dΩ)
    avg_τ_f = sum(∫(τ_f)dΩ) / sum(∫(1.0)dΩ)
    
    @info "Velocity field statistics:"
    @info "  Avg |v_ρ| = $avg_v_ρ"
    @info "  Avg |v_f| = $avg_v_f"
    @info "  Avg τ_ρ = $avg_τ_ρ"
    @info "  Avg τ_f = $avg_τ_f"
    
    return (
        velocity_leader=v_ρ,
        velocity_follower=v_f,
        velocity_leader_mag=v_ρ_mag,
        velocity_follower_mag=v_f_mag,
        tau_leader=τ_ρ,
        tau_follower=τ_f,
        avg_velocity_leader=avg_v_ρ,
        avg_velocity_follower=avg_v_f,
        avg_tau_leader=avg_τ_ρ,
        avg_tau_follower=avg_τ_f
    )
end

# =============================================================================
# Main Execution
# =============================================================================

if abspath(PROGRAM_FILE) == @__FILE__
    @info "Running Cancer Invasion Test with SUPG"
    
    # Test 1: Standard SUPG (C=0.5)
    params = CancerInvasionSUPG(
        α=1.0, a=0.44,
        δ=1.377, d=1.377,
        γ=0.6048, g=1.1772,
        ηy=10.0,
        C=0.5,              # SUPG constant
        domain=(0.0, 1.0, 0.0, 1.0),
        partition=(20, 20), # Smaller for faster testing
        tspan=(0.0, 0.1),   # Short time for testing
        reltol=1e-3,
        abstol=1e-5,
        dtmax=0.01
    )
    
    @info ""
    @info "Test 1: SUPG with C=$(params.C)"
    
    # Solve with SUPG
    sol, model, Ω, dΩ, U, N, C_supg = solve_supg(params)
    
    # Check results
    diagnostics = check_solution(sol, model, Ω, dΩ, U, N, C_supg)
    
    # SUPG-specific diagnostics
    supg_diag = get_supg_diagnostics(sol, model, Ω, dΩ, U, N, params)
    
    # Test 2: Verify against original (C=0, should match non-SUPG)
    @info ""
    @info "="^70
    @info "Test 2: Verification with C=0 (should match standard Galerkin)"
    @info "="^70
    
    params_verify = CancerInvasionSUPG(
        α=1.0, a=0.44,
        δ=1.377, d=1.377,
        γ=0.6048, g=1.1772,
        ηy=10.0,
        C=0.0,              # No SUPG
        domain=(0.0, 1.0, 0.0, 1.0),
        partition=(20, 20),
        tspan=(0.0, 0.1),
        reltol=1e-3,
        abstol=1e-5,
        dtmax=0.01
    )
    
    sol_verify, _, _, _, _, _, _ = solve_supg(params_verify, verbose=false)
    
    # Compare final states
    u_final_supg = sol.u[end]
    u_final_nosupg = sol_verify.u[end]
    
    error = norm(u_final_supg - u_final_nosupg) / norm(u_final_nosupg)
    @info ""
    @info "Verification C=$(params.C) vs C=0:"
    @info "  Relative difference: $error"
    @info "  (C=0.5 should show differences from C=0)"
    
    @info ""
    @info "All tests complete!"
end
