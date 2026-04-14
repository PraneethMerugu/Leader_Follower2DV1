# Technology Stack: Gridap AMR for Cancer Invasion

**Project:** Cancer Invasion PDE with SUPG Stabilization and AMR
**Researched:** April 2025

## Recommended Stack

### Core Framework

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| Julia | 1.9+ | Language | Gridap ecosystem requires Julia 1.9+ for best compatibility |
| Gridap.jl | 0.19+ | Core FE framework | Primary FEM library with AMR support |
| GridapP4est.jl | 0.3+ | Parallel AMR | Octree/quadtree AMR via p4est library |
| GridapDistributed.jl | 0.4+ | Distributed meshes | MPI-parallel mesh management |
| PartitionedArrays.jl | 0.3+ | Parallel arrays | Data distribution for MPI |
| EquilibratedFlux.jl | latest | Error estimation | A posteriori error estimators for AMR |

### Build Dependencies

| Technology | Purpose | Notes |
|------------|---------|-------|
| MPI.jl | MPI bindings | Required for GridapDistributed |
| P4est_wrapper.jl | p4est bindings | C library wrapper for octree meshes |
| sc | libsc library | p4est dependency (system library) |
| p4est | p4est library | Octree mesh library (system library) |

### Time Integration (Sundials IDA Replacement)

| Technology | Purpose | Why |
|------------|---------|-----|
| Gridap.ODEs | ODE solvers | Native Gridap transient support |
| Rodas5 | Rosenbrock method | Stiff problems, no algebraic constraints |
| GeneralizedAlpha2 | Implicit solver | Alternative for structural dynamics |
| OrdinaryDiffEq.jl | Fallback | If Gridap ODEs insufficient |

### Supporting Libraries

| Library | Purpose | When to Use |
|---------|---------|-------------|
| FillArrays.jl | Array operations | Gridap dependency |
| LinearAlgebra | Sparse solvers | Required for assembly |
| SparseArrays.jl | Sparse matrices | Gridap dependency |
| NLsolve.jl | Nonlinear solves | For nonlinear PDEs |
| ForwardDiff.jl | Autodiff | For Jacobian computation |

## Installation

### System Dependencies (macOS/Linux)

```bash
# macOS with Homebrew
brew install p4est sc

# Ubuntu/Debian
sudo apt-get install libp4est-dev libsc-dev

# From source (if packages unavailable)
# See: https://github.com/gridap/p4est_wrapper.jl
```

### Julia Package Installation

```julia
using Pkg

# Core AMR packages
Pkg.add("Gridap")
Pkg.add("GridapP4est")
Pkg.add("GridapDistributed")
Pkg.add("PartitionedArrays")

# Error estimation
Pkg.add(url="https://github.com/aerappa/EquilibratedFlux.jl")

# Time integration
Pkg.add("Gridap")  # ODEs module included

# MPI support
Pkg.add("MPI")
Pkg.add("MPIPreferences")
using MPIPreferences
MPIPreferences.use_system_binary()  # Use system MPI
```

### Verification

```julia
using Gridap
using GridapP4est
using GridapDistributed
using PartitionedArrays
using EquilibratedFlux

# Test GridapP4est
MPI.Init()
GridapP4est.with(MPI.COMM_WORLD) do
    coarse_model = CartesianDiscreteModel((0,1,0,1), (2,2))
    dmodel = UniformlyRefinedForestOfOctreesDiscreteModel(MPI.COMM_WORLD, coarse_model, 2)
    println("GridapP4est working: $(num_cells(dmodel)) cells")
end

# Test EquilibratedFlux
model = CartesianDiscreteModel((0,1,0,1), (10,10)) |> simplexify
reffe = ReferenceFE(lagrangian, Float64, 1)
V = TestFESpace(model, reffe; conformity=:H1, dirichlet_tags="boundary")
uh = interpolate(x->sin(2π*x[1])*sin(2π*x[2]), V)
σ = build_averaged_flux(∇(uh), model)
println("EquilibratedFlux working")
```

## Alternatives Considered

| Category | Recommended | Alternative | Why Not |
|----------|-------------|-------------|---------|
| AMR Backend | GridapP4est.jl | Gridap.Adaptivity (serial) | Need distributed parallel for cancer scale |
| Error Estimator | EquilibratedFlux.jl | Residual-based estimators | Equilibrated flux is more accurate, reliable |
| Time Integrator | Rodas5 (Gridap.ODEs) | Sundials IDA | IDA requires DAE formulation; Rodas5 simpler for stiff ODEs |
| Mesh Library | p4est (via GridapP4est) | libMesh | p4est scales better, better Julia integration |

## Stack Compatibility Notes

### Version Constraints

```toml
# Project.toml constraints
[compat]
julia = "1.9,1.10,1.11"
Gridap = "0.19.9"
GridapP4est = "0.3.13"
GridapDistributed = "0.4.13"
PartitionedArrays = "0.3.3"
MPI = "0.20"
```

### Known Issues

1. **macOS ARM (M1/M2/M3)**: p4est may need compilation from source
2. **Windows**: p4est support limited; use WSL2
3. **EquilibratedFlux**: Currently triangular meshes only (TRI), quads need special handling
4. **MPI Version**: Must use consistent MPI across all packages

## Hardware Requirements

| Use Case | CPU | Memory | MPI Ranks |
|----------|-----|--------|-----------|
| Development | 4+ cores | 16GB | 1-4 |
| Small tests | 8+ cores | 32GB | 4-8 |
| Production | 64+ cores | 256GB+ | 16-64 |

## Sources

- GridapP4est.jl Project.toml: https://github.com/gridap/GridapP4est.jl/blob/main/Project.toml
- Gridap.jl ODEs module: https://github.com/gridap/Gridap.jl/tree/master/src/ODEs
- EquilibratedFlux.jl: https://github.com/aerappa/EquilibratedFlux.jl
- p4est_wrapper.jl: https://github.com/gridap/p4est_wrapper.jl
