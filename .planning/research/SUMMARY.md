# Research Summary: Gridap AMR Ecosystem for Cancer Invasion PDE

**Domain:** Adaptive Mesh Refinement (AMR) with Gridap.jl for transient PDEs
**Researched:** April 2025
**Overall confidence:** HIGH for GridapP4est, MEDIUM for EquilibratedFlux integration

## Executive Summary

This research investigates the Gridap ecosystem's adaptive mesh refinement (AMR) capabilities for a cancer invasion PDE model using SUPG stabilization. The primary finding is that **GridapP4est.jl** provides robust octree/quadtree-based AMR with distributed memory support, while **EquilibratedFlux.jl** offers a posteriori error estimation. For transient problems, Gridap's native ODE solvers can replace Sundials IDA with Rosenbrock methods (Rodas5), though the integration of AMR with transient solvers requires careful handling of solution transfer between refined meshes.

The recommended architecture involves:
1. **GridapP4est.jl** for parallel AMR mesh management
2. **EquilibratedFlux.jl** for goal-oriented error indicators
3. **Gridap.ODEs** module with Rodas5 for time integration
4. Custom solution transfer operators between adaptive timesteps

## Key Findings

**Stack:** Gridap.jl + GridapP4est.jl + EquilibratedFlux.jl + GridapDistributed.jl + PartitionedArrays.jl

**Architecture:** Parallel distributed AMR with non-conforming hanging node constraints, equilibrated flux error estimation, and Rosenbrock time integration

**Critical pitfall:** Solution transfer between differently refined meshes in transient simulations requires explicit handling through Gridap's `AdaptivityGlue` and `FineToCoarseFields` mechanisms

## Implications for Roadmap

### Suggested Phase Structure:

1. **Phase 1: Serial AMR Setup** - Core GridapP4est integration
   - Addresses: Octree mesh setup, refinement/coarsening API
   - Avoids: Distributed memory complexity initially

2. **Phase 2: Error Estimation** - EquilibratedFlux integration
   - Addresses: Error indicator computation, Dörfler marking
   - Avoids: Goal-oriented adaptation complexity initially

3. **Phase 3: Transient + AMR** - Time-dependent adaptation
   - Addresses: Rodas5 integration, solution transfer
   - Avoids: Complex IMEX operators initially

4. **Phase 4: Distributed Parallel** - Full MPI scaling
   - Addresses: GridapDistributed, PartitionedArrays
   - Avoids: Load balancing complexity initially

5. **Phase 5: Goal-Oriented AMR** - Cancer-specific adaptation
   - Addresses: Tumor boundary refinement, SUPG stabilization
   - Avoids: Anisotropic refinement initially

### Phase Ordering Rationale:
- Serial AMR must precede parallel AMR
- Error estimation must precede goal-oriented adaptation
- Stationary AMR must precede transient AMR (solution transfer complexity)
- Basic transient must precede SUPG stabilization (stiffness handling)

### Research Flags for Phases:
- **Phase 3 (Transient + AMR)**: HIGH - Solution transfer operators need validation
- **Phase 5 (Goal-Oriented)**: MEDIUM - EquilibratedFlux API integration patterns
- **Phase 4 (Distributed)**: LOW - Well-documented in GridapP4est tests

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| GridapP4est Stack | HIGH | Official package, extensive tests, stable API |
| EquilibratedFlux | MEDIUM | Third-party but well-documented, focused scope |
| Transient + AMR | MEDIUM | Gridap ODEs module mature, but AMR+transient examples limited |
| Solution Transfer | MEDIUM | Gridap.Adaptivity provides glue structures, need validation |
| SUPG + AMR | LOW | No specific examples found, theoretical validation needed |

## Gaps to Address

1. **Transient AMR Integration**: GridapP4est examples focus on stationary problems; transient AMR requires custom solution transfer
2. **SUPG Stabilization**: Need to verify SUPG works with hanging nodes and non-conforming meshes
3. **Goal-Oriented Error Estimators**: EquilibratedFlux provides flux reconstruction; goal-oriented adaptation needs dual problem setup
4. **Load Balancing**: Dynamic repartitioning after refinement not automatic
5. **Checkpoint/Restart**: No evidence of native checkpointing with AMR meshes

## Key Sources

- GridapP4est.jl GitHub: https://github.com/gridap/GridapP4est.jl
- EquilibratedFlux.jl GitHub: https://github.com/aerappa/EquilibratedFlux.jl
- Gridap ODEs Documentation: https://gridap.github.io/Gridap.jl/dev/modules/ODEs/
- Gridap Adaptivity Documentation: https://gridap.github.io/Gridap.jl/dev/modules/Adaptivity/
- Gridap Tutorial 21 (Poisson with AMR): https://gridap.github.io/Tutorials/dev/pages/t021_poisson_amr/
