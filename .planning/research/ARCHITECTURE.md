# Architecture Patterns: Gridap AMR for Cancer Invasion

**Domain:** Adaptive mesh refinement with transient PDEs
**Researched:** April 2025

## Recommended Architecture

### High-Level Component Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    Cancer Invasion Solver                    │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │   Cancer     │  │   SUPG       │  │   AMR        │       │
│  │   Model      │  │   Stabilize  │  │   Driver     │       │
│  │  (reaction-  │  │              │  │              │       │
│  │   diffusion) │  │              │  │              │       │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘       │
│         │                 │                 │                │
│         └─────────────────┼─────────────────┘                │
│                         │                                   │
│  ┌───────────────────────┴───────────────────────┐          │
│  │         TransientFEOperator (Gridap.ODEs)    │          │
│  │              ┌──────────────────┐            │          │
│  │              │  ODE Solver      │            │          │
│  │              │  (Rodas5/IMEX)   │            │          │
│  │              └──────────────────┘            │          │
│  └───────────────────────────────────────────────┘          │
│                         │                                   │
│  ┌───────────────────────┴───────────────────────┐          │
│  │         GridapP4est Distributed Model         │          │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐     │          │
│  │  │  Octree  │  │ Hanging  │  │ Solution │     │          │
│  │  │  Mesh    │  │ Node     │  │ Transfer │     │          │
│  │  │          │  │ Constraints│ │          │     │          │
│  │  └──────────┘  └──────────┘  └──────────┘     │          │
│  └───────────────────────────────────────────────┘          │
│                         │                                   │
│  ┌───────────────────────┴───────────────────────┐          │
│  │      EquilibratedFlux Error Estimator        │          │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐   │          │
│  │  │  Flux    │  │  Error   │  │  Marking │   │          │
│  │  │  Recon   │  │  Compute │  │  Strategy│   │          │
│  │  └──────────┘  └──────────┘  └──────────┘   │          │
│  └───────────────────────────────────────────┘          │
└─────────────────────────────────────────────────────────────┘
```

### Component Boundaries

| Component | Responsibility | Communicates With |
|-----------|---------------|-------------------|
| Cancer Model | Define PDE weak form, boundary conditions | TransientFEOperator |
| SUPG Stabilization | Add stabilization terms for advection | Cancer Model |
| AMR Driver | Orchestrate refinement, marking, adaptation | GridapP4est, Error Estimator |
| TransientFEOperator | Wrap PDE as ODE system | ODE Solver, FE Spaces |
| ODE Solver (Rodas5) | Time integration, adaptive timestep | TransientFEOperator |
| GridapP4est | Distributed octree mesh management | FE Spaces, Solution Transfer |
| Hanging Node Constraints | Enforce conformity on non-conforming mesh | FESpace construction |
| Solution Transfer | Interpolate solutions between meshes | AMR Driver, ODE Solver |
| EquilibratedFlux | Compute error estimators cell-wise | AMR Driver |
| Marking Strategy | Select cells for refinement/coarsening | AMR Driver |

## Data Flow

### Transient AMR Workflow

```
Initialize
    │
    ▼
┌─────────────────┐
│ Create Octree   │◄── GridapP4est
│ Mesh (coarse)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Setup FE Spaces │◄── With hanging node constraints
│ (trial/test)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Define Transient│
│ FE Operator     │◄── Cancer model + SUPG
└────────┬────────┘
         │
    Time Loop
         │
         ▼
┌─────────────────┐
│ Solve Timestep  │◄── Rodas5 ODE solver
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Check Adapt     │
│ Criteria        │
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
    ▼         │
┌────────┐    │
│ Refine │    │ No refinement
│ Mesh   │    │
└───┬────┘    │
    │         │
    ▼         │
┌────────┐    │
│Transfer│    │
│Solution│    │
└───┬────┘    │
    │         │
    ▼         ▼
┌──────────────┐
│ Update FE    │
│ Spaces       │
└──────┬───────┘
       │
       ▼
   Continue
   Time Loop
```

### Solution Transfer Pattern

```julia
# After AMR, transfer solution to new mesh
function transfer_solution(uh_old, model_old, model_new, glue)
    # Method 1: Using Gridap's built-in transfer
    uh_new = interpolate(FineToCoarseField(uh_old, glue), Vh_new)
    
    # Method 2: Manual interpolation (if needed)
    # uh_new = interpolate(uh_old ∘ inverse_map(glue), Vh_new)
    
    return uh_new
end
```

## Patterns to Follow

### Pattern 1: Hanging Node Constraint Handling

**What:** When using AMR with non-conforming meshes, hanging nodes (T-junctions) need constraints to maintain continuity.

**When:** Always with GridapP4est (automatic)

**Example:**
```julia
# GridapP4est automatically adds constraints
model = OctreeDistributedDiscreteModel(parts, coarse_model, num_uniform_refinements)

# FESpace constructor handles constraints automatically
V = FESpace(model, reffe; conformity=:H1, dirichlet_tags="boundary")
# Returns FESpaceWithLinearConstraints if hanging nodes present
```

### Pattern 2: Adaptivity Loop

**What:** Standard solve → estimate → mark → refine workflow.

**When:** Every adaptation step

**Example:**
```julia
function adaptivity_step(model, uh, Vh, strategy)
    # 1. ESTIMATE
    error_indicators = compute_error_indicators(uh, model)
    
    # 2. MARK
    flags = compute_refinement_flags(error_indicators, strategy)
    
    # 3. REFINE
    model_new, glue = adapt(model, flags)
    
    # 4. TRANSFER
    Vh_new = FESpace(model_new, reffe; conformity=:H1, dirichlet_tags="boundary")
    uh_new = interpolate(FineToCoarseField(uh, glue), Vh_new)
    
    return model_new, Vh_new, uh_new, glue
end
```

### Pattern 3: Transient AMR Integration

**What:** Coordinate AMR with time stepping, preserving solution continuity.

**When:** Transient problems with mesh adaptation

**Example:**
```julia
# Time stepping with periodic AMR
t = t0
uh = uh0
while t < tF
    # Adapt mesh periodically
    if should_adapt(t, step)
        model, Vh, uh, _ = adaptivity_step(model, uh, Vh, strategy)
        # Update operator with new spaces
        op = TransientFEOperator(res, jacs, U(Vh), V(Vh))
    end
    
    # Take timestep
    dt = compute_timestep(op, uh, t)
    uh = ode_step(odesolver, op, uh, t, dt)
    t += dt
end
```

### Pattern 4: Parallel Communication Pattern

**What:** MPI-parallel mesh and solution management.

**When:** Distributed runs with GridapP4est

**Example:**
```julia
using PartitionedArrays

# Initialize MPI and partitioned arrays
ranks = distribute(LinearIndices((MPI.Comm_size(MPI.COMM_WORLD),)))

# All operations are distributed automatically
GridapP4est.with(ranks) do
    model = OctreeDistributedDiscreteModel(ranks, coarse_model, refinements)
    # ... FE setup and solve ...
end
```

## Anti-Patterns to Avoid

### Anti-Pattern 1: Manual Constraint Handling

**What:** Trying to manually set up hanging node constraints.

**Why bad:** GridapP4est does this automatically; manual setup leads to errors.

**Instead:** Use `FESpace(model, reffe; ...)` with GridapP4est models.

### Anti-Pattern 2: Global Mesh Rebuild

**What:** Rebuilding entire mesh structure on each AMR step.

**Why bad:** Expensive, loses all parallel distribution info.

**Instead:** Use `adapt()` which incrementally updates mesh.

### Anti-Pattern 3: Ignoring Ghost Cells

**What:** Only working with owned cells in parallel.

**Why bad:** Error estimators need ghost cell contributions; solution transfer needs ghost data.

**Instead:** Always use `with_ghost` triangulations:
```julia
Ω = Triangulation(with_ghost, model)  # Include ghost cells
dΩ = Measure(Ω, degree)
```

### Anti-Pattern 4: Solution Transfer Without Glue

**What:** Manually interpolating without using `AdaptivityGlue`.

**Why bad:** Loses parent-child relationships, incorrect on hanging nodes.

**Instead:** Use `FineToCoarseField` or `OldToNewField` with glue.

## Scalability Considerations

| Concern | At 100 cells | At 10K cells | At 1M cells |
|---------|--------------|--------------|-------------|
| Mesh Storage | Single array | Distributed | Distributed + p4est |
| AMR Frequency | Every step | Every 5 steps | Every 10-100 steps |
| Error Estimation | Cell-wise serial | Cell-wise parallel | Patch-based parallel |
| Load Balancing | Not needed | Periodic | After each AMR |
| Solution Transfer | Direct | With ghosts | Parallel exchange |

## Integration Code Snippets

### Complete Minimal Example: Transient AMR

```julia
using Gridap
using GridapP4est
using GridapDistributed
using PartitionedArrays
using EquilibratedFlux
using MPI

function transient_amr_driver(parts, coarse_model, t0, tF)
    # Initialize p4est
    GridapP4est.with(parts) do
        # Initial uniform refinement
        model = OctreeDistributedDiscreteModel(parts, coarse_model, 2)
        
        # FE spaces
        order = 1
        reffe = ReferenceFE(lagrangian, Float64, order)
        V = TestFESpace(model, reffe; conformity=:H1, dirichlet_tags="boundary")
        U = TrialFESpace(V)
        
        # Initial solution
        uh = interpolate(x->0.0, U)
        
        # Time stepping
        dt = 0.01
        t = t0
        while t < tF
            # Solve timestep (simplified - use proper ODE solver)
            uh = solve_timestep(model, U, V, uh, dt)
            
            # Periodic AMR
            if should_adapt(t)
                # Estimate error
                σ = build_equilibrated_flux(-∇(uh), f, model, order)
                error_indicators = compute_cell_errors(σ, uh)
                
                # Mark and adapt
                strategy = FixedFractionAdaptiveFlagsMarkingStrategy(0.2, 0.05)
                flags = compute_refinement_flags(error_indicators, strategy)
                model_new, glue = adapt(model, flags)
                
                # Transfer solution
                V_new = TestFESpace(model_new, reffe; conformity=:H1)
                U_new = TrialFESpace(V_new)
                uh = interpolate(FineToCoarseField(uh, glue), U_new)
                
                # Update
                model = model_new
                V = V_new
                U = U_new
            end
            
            t += dt
        end
        
        return uh
    end
end
```

## Sources

- GridapP4est.jl source: https://github.com/gridap/GridapP4est.jl/tree/main/src
- Gridap Adaptivity module: https://github.com/gridap/Gridap.jl/tree/master/src/Adaptivity
- Gridap ODEs module: https://github.com/gridap/Gridap.jl/tree/master/src/ODEs
- EquilibratedFlux.jl: https://github.com/aerappa/EquilibratedFlux.jl
