# Argos.jl

[![][docs-stable-img]][docs-stable-url] [![][build-latest-img]][build-url] [![][codecov-latest-img]][codecov-latest-url]

Argos.jl extends the power-system modeler [ExaPF.jl](https://github.com/exanauts/ExaPF.jl)
and the interior-point solver [MadNLP.jl](https://github.com/MadNLP/MadNLP.jl)
to solve the optimal power flow (OPF) problem entirely in Julia.

The package is structured as follows:
- in `src/Evaluators/`, various optimization evaluators implement the different callbacks (objective, gradient, Hessian)
  required in the optimization algorithms .
- in `src/Algorithms/`, an Augmented Lagrangian algorithm is implemented, targeting
  primarily the resolution of large-scale OPF problems on GPU architectures.
- in `src/Wrappers/`, a wrapper for [MathOptInterface](https://github.com/jump-dev/MathOptInterface.jl) and a wrapper for [NLPModels.jl](https://github.com/JuliaSmoothOptimizers/NLPModels.jl/) are implemented.

## Installation

Argos.jl is currently unregistered. To install it, enter in the REPL the command:
```julia
add "https://github.com/exanauts/Argos.jl"
```

To check that everything is working as expected, please run
```julia
test Argos
```

By default, this command tests all the `Evaluators` implemented in Argos
on the CPU and, if available, on a CUDA GPU.

## Quickstart

Once Argos installed, we can use the function `run_opf` to solve
the OPF with MadNLP. The function takes as input any MATPOWER file:
```julia
# Solve in the full-space
ips = Argos.run_opf("data/case9.m", Argos.FullSpace())

```
The second argument specifies the formulation used inside MadNLP to solve
the OPF problem. `FullSpace()` implements the classical full-space formulation,
(as implemented inside MATPOWER or PowerModels.jl). Alternatively, one may want to solve
the OPF using the reduced-space formulation of Dommel and Tinney:
```julia
# Solve in the reduced-space
ips = Argos.run_opf("data/case9.m", Argos.DommelTinney())

```

## How to use Argos' evaluators?

Argos implements two evaluators to solve the OPF problem:
the `FullSpaceEvaluator` implements the classical OPF formulation
in the full-space, whereas `ReducedSpaceEvaluator` implements the
reduced-space formulation of Dommel & Tinney.

### Using an evaluator
Instantiating a new evaluator from a MATPOWER file simply amounts to
```julia
# Reduced-space evaluator
nlp = Argos.ReducedSpaceEvaluator("case57.m")
# Full-space evaluator
flp = Argos.FullSpaceEvaluator("case57.m")
```

An initial optimization variable can be computed as
```julia
u = Argos.initial(nlp)
```
The variable `u` is the control that will be used all throughout the
optimization. Once a new point `u` obtained, one can refresh all the structures
inside `nlp` with:
```julia
Argos.update!(nlp, u)
```
Once the structures refreshed, the other callbacks can be evaluated as well:
```julia
Argos.objective(nlp, u) # objective
Argos.gradient(nlp, u)  # reduced gradient
Argos.jacobian(nlp, u)  # reduced Jacobian
Argos.hessian(nlp, u)   # reduced Hessian
```


### MOI wrapper

Argos implements a wrapper to [MathOptInterface](https://github.com/jump-dev/MathOptInterface.jl)
to solve the optimal power flow problem with any nonlinear optimization solver compatible
with MathOptInterface:
```julia
nlp = Argos.ReducedSpaceEvaluator("case57.m")
optimizer = Ipopt.Optimizer() # MOI optimizer
# Update tolerance to be above tolerance of Newton-Raphson subsolver
MOI.set(optimizer, MOI.RawOptimizerAttribute("tol"), 1e-5)
# Solve reduced space problem
solution = Argos.optimize!(optimizer, nlp)
```

### NLPModels wrapper

Alternatively, one can use NLPModels.jl to wrap any evaluators implemented
in Argos. This amounts simply to:
```julia
nlp = Argos.FullSpaceEvaluator("case57.m")
# Wrap in NLPModels
model = Argos.OPFModel(nlp)

x0 = NLPModels.get_x0(model)
obj = NLPModels.obj(model, x0)

```
Once the evaluator wrapped inside NLPModels, we can leverage any
solver implemented in [JuliaSmoothOptimizers](https://github.com/JuliaSmoothOptimizers/)
to solve the OPF problem.


## How to deport the solution of the OPF on the GPU?
[`ExaPF.jl`](https://github.com/exanauts/ExaPF-Opt.jl) is
using [`KernelAbstractions`](https://github.com/JuliaGPU/KernelAbstractions.jl)
to implement all its core operations. Hence, deporting the computation
on GPU accelerators is straightforward. Argos.jl inherits this behavior and
all evaluators can be instantiated on GPU accelerators, simply as
```julia
using CUDAKernels # Load CUDA backend for KernelAbstractions
using ArgosCUDA
nlp = Argos.ReducedSpaceEvaluator("case57.m"; device=CUDADevice())
```
When doing so, all kernels are instantiated on the GPU to avoid
memory transfer between the host and the device. The sparse linear
algebra operations are handled by `cuSPARSE`, and the sparse factorizations
are performed using `cusolverRF`. When deporting the computation on the
GPU, the reduced Hessian can be evaluated in parallel.


### Batch evaluation of the reduced Hessian
Instead of computing the reduced Hessian one Hessian-vector product after one Hessian-vector product,
the Hessian-vector products are directly evaluated in batch in this case.
To activate the batch evaluation for the reduced Hessian, please specify
the number of Hessian-vector products to perform in one batch as
```julia
nlp = Argos.ReducedSpaceEvaluator("case57.m"; device=CUDADevice(), nbatch_hessian=8)
```
Note that on large instances, the batch computation could be quite heavy on the
GPU's memory.

[docs-latest-img]: https://img.shields.io/badge/docs-latest-blue.svg
[docs-latest-url]: https://exanauts.github.io/Argos/
[build-url]: https://github.com/exanauts/Argos.jl/actions?query=workflow
[build-latest-img]: https://github.com/exanauts/Argos.jl/workflows/Run%20tests/badge.svg?branch=master
[codecov-latest-img]: https://codecov.io/gh/exanauts/Argos.jl/branch/master/graphs/badge.svg?branch=master
[codecov-latest-url]: https://codecov.io/github/exanauts/Argos.jl?branch=master
