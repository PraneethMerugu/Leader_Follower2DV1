# Feature Landscape: Gridap AMR for Cancer Invasion

**Domain:** Adaptive mesh refinement for reaction-diffusion-advection PDEs
**Researched:** April 2025

## Table Stakes

Features users expect in an AMR-capable cancer invasion solver.

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| Octree/Quadtree AMR | Standard for 2D/3D adaptivity | Medium | GridapP4est provides via p4est |
| Hanging Node Constraints | Non-conforming mesh conformity | High | GridapP4est handles with linear constraints |
| Error-based Refinement | AMR driven by solution error | Medium | EquilibratedFlux.jl provides |
| Parallel Distributed Mesh | Cancer problems need scale | Medium | GridapDistributed.jl + MPI |
| Solution Transfer | Interpolate between meshes | Medium | Gridap.Adaptivity provides |
| Ghost Layers | Parallel communication | Low | Automatic in GridapP4est |
| Refinement/Coarsening | Both directions needed | Medium | Both supported in GridapP4est |
| Mesh I/O | Save/load adapted meshes | Low | VTK/JSON output available |

## Differentiators

Features that set this implementation apart.

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| Equilibrated Flux Estimation | Guaranteed upper bound on error | High | More reliable than residual estimators |
| Goal-Oriented AMR | Focus resources on tumor boundary | High | Need dual problem setup |
| SUPG Stabilization | Handle advection-dominated regimes | Medium | Standard for cancer invasion |
| Transient AMR | Adapt mesh as solution evolves | High | Requires careful transfer |
| Anisotropic Refinement | Refine in specific directions | High | Vertical/horizontal in GridapP4est |
| Automatic Load Balancing | Maintain efficiency during AMR | Medium | Available in p4est |

## Anti-Features

Features to explicitly NOT build.

| Anti-Feature | Why Avoid | What to Do Instead |
|--------------|-----------|-------------------|
| Serial-only AMR | Cancer problems need scale | Start serial, move to distributed |
| h-refinement only | May miss optimal convergence | Support coarsening from start |
| Global mesh reconstruction | Too expensive per step | Incremental AMR with local operations |
| Manual constraint handling | Error-prone | Use GridapP4est's built-in constraints |
| Custom mesh library | Maintenance burden | Use battle-tested p4est via GridapP4est |

## Feature Dependencies

```
Octree AMR
  ├─ Hanging Node Constraints
  │   └─ Linear Constraint Solver
  ├─ Error Estimation
  │   ├─ Residual-based (simple)
  │   └─ Equilibrated Flux (accurate)
  │       └─ Patch-based Local Solves
  ├─ Marking Strategy
  │   ├─ Fixed Fraction
  │   └─ Dörfler Marking
  ├─ Refinement/Coarsening
  │   └─ Solution Transfer
  └─ Parallel Distribution (optional)
      ├─ Ghost Layers
      └─ Load Balancing

Transient Solver
  ├─ Time Stepping
  │   ├─ Implicit (Rodas5)
  │   └─ IMEX (for advection)
  ├─ SUPG Stabilization
  │   └─ Stabilization Parameters
  └─ AMR Integration
      ├─ Temporal Error Estimation
      ├─ Mesh Update Frequency
      └─ Solution Interpolation
```

## Key API Functions by Package

### GridapP4est.jl

```julia
# Core constructors
UniformlyRefinedForestOfOctreesDiscreteModel(parts, coarse_model, num_refinements)
OctreeDistributedDiscreteModel(parts, coarse_model, num_refinements)
AnisotropicallyAdapted3DDistributedDiscreteModel(parts, coarse_2d_model, h_refine, v_refine)

# Adaptivity
adapt(model, refinement_flags)  # Returns (new_model, adaptivity_glue)
vertically_adapt(model, flags)
horizontally_adapt(model, flags)

# Marking
FixedFractionAdaptiveFlagsMarkingStrategy(refine_frac, coarsen_frac)
update_adaptivity_flags!(flags, strategy, cell_partition, error_indicators)

# Flags
refine_flag, coarsen_flag, nothing_flag
```

### EquilibratedFlux.jl

```julia
# Error estimators
build_equilibrated_flux(-∇(uh), f, model, order)  # σ_eq
build_averaged_flux(∇(uh), model)                   # σ_ave

# Error computation
η² = L2_norm_squared(σ_eq + ∇(uh), dx)
η_arr = sqrt.(getindex(η², trian))

# Marking
cells_to_refine = dorfler_marking(η_arr, θ=0.3)
```

### Gridap.Adaptivity

```julia
# Refinement
refine(model; refinement_method="nvb", cells_to_refine=indices)

# Adaptivity glue
AdaptivityGlue(fine_to_coarse_faces_map, child_ids, refinement_rules)
get_n2o_reference_coordinate_map(glue)

# Solution transfer
FineToCoarseField(uh_fine, glue)
OldToNewField(uh_old, glue)
```

### Gridap.ODEs

```julia
# ODE operators
TransientFEOperator(res, jacs, trial, test)
TransientQuasilinearFEOperator(a, b, trial, test)

# Solvers
ThetaMethod(odeslvr, dt, θ)  # θ=0.5: Crank-Nicolson, θ=1.0: Backward Euler
RungeKutta(odeslvr, tableau)  # Includes Rodas5
GeneralizedAlpha2(ρ∞, dt)

# Transient solution
tfesltn = solve(odeslvr, tfeop, t0, tF, uh0)
for (t_n, uh_n) in tfesltn
    # Solution at time t_n
end
```

## MVP Recommendation

### Prioritize:

1. **Stationary AMR with Error Estimation**
   - Solve Laplace on L-shaped domain
   - Equilibrated flux error estimator
   - Dörfler marking
   - Demonstrate convergence

2. **Simple Transient Problem**
   - Heat equation on fixed mesh
   - Rodas5 time integration
   - Verify correctness

3. **Combined Transient + AMR**
   - Heat equation with AMR
   - Solution transfer at each adaptation
   - Validate against reference solution

### Defer:

- **SUPG Stabilization**: Until basic transient AMR works
- **Goal-Oriented AMR**: Until error estimation validated
- **Distributed Parallel**: Until serial version robust
- **Anisotropic Refinement**: Until isotropic works
- **Complex Cancer Model**: Until infrastructure validated

## Sources

- GridapP4est test files: https://github.com/gridap/GridapP4est.jl/tree/main/test
- EquilibratedFlux examples: https://github.com/aerappa/EquilibratedFlux.jl/tree/main/doc/examples
- Gridap Tutorial 21 (AMR): https://gridap.github.io/Tutorials/dev/pages/t021_poisson_amr/
- Gridap Tutorial 17 (Transient): https://gridap.github.io/Tutorials/dev/pages/t017_transient_linear/
