# ExaOpt

[![][docs-latest-img]][docs-latest-url] ![CI](https://github.com/exanauts/ExaPF-Opt/workflows/Run%20tests/badge.svg?branch=master)

[docs-latest-img]: https://img.shields.io/badge/docs-latest-blue.svg
[docs-latest-url]: https://exanauts.github.io/ExaOpt/

ExaOpt extends [ExaPF.jl](https://github.com/exanauts/ExaPF.jl) by implementing
the optimization routines required to solve the optimal power flow (OPF) problems.
The routines come in two flavors:
- in `src/Evaluators/`, various optimization evaluators implement the different callbacks
  required in the optimization algorithm (objective, gradient, Hessian).
- in `src/Algorithms/`, an Augmented Lagrangian algorithm is implemented, targeting
  primarily the resolution of large-scale OPF problems on GPU architectures.

## Installation

ExaOpt is currently unregistered. To install it, you should be a member
of the [*exanauts*](https://exanauts.github.io/) organization.

```julia
add "git@github.com:exanauts/ExaPF-Opt.git"
```

To check that everything is working as expected, please run
```julia
test ExaOpt
```

By default, this command tests all the `Evaluators` implemented in ExaOpt
on the CPU and, if available, on CUDA GPUs.

## Quickstart

ExaOpt's cornerstone is the `ReducedSpaceEvaluator`, allowing to run
the optimization algorithm directly in the reduced space induced by
the power flow equations. Instantiating a new `ReducedSpaceEvaluator`
from a MATPOWER file simply amounts to
```julia
nlp = ExaOpt.ReducedSpaceEvaluator("case57.m")
```

An initial optimization variable can be computed as
```julia
u = ExaPF.initial(nlp)
```
The variable `u` is the control that will be used all throughout the
optimization. Then, the function `update!` solves the power flow and updates accordingly all the structures
inside `nlp`:
```julia
ExaPF.update!(nlp, u)
```
Once the power flow equations solved, the other callbacks can be evaluated
as well:
```julia
ExaPF.objective(nlp, u) # objective
ExaPF.gradient(nlp, u)  # reduced gradient
ExaPF.jacobian(nlp, u)  # reduced Jacobian
ExaPF.hessian(nlp, u)   # reduced Hessian
```

### Use the ReducedSpaceEvaluator on GPU accelerators
[`ExaPF.jl`](https://github.com/exanauts/ExaPF-Opt.jl) is
using [`KernelAbstractions`](https://github.com/JuliaGPU/KernelAbstractions.jl)
to implement all its core operations. Hence, deporting the computation in ExaPF
on GPU accelerators is straightforward. ExaOpt inherits this behavior and
all evaluators can be instantiated on GPU accelerators, simply as
```julia
using CUDAKernels # Load CUDA backend for KernelAbstractions
nlp = ExaOpt.ReducedSpaceEvaluator("case57.m"; device=CUDADevice())
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
nlp = ExaOpt.ReducedSpaceEvaluator("case57.m"; device=CUDADevice(), nbatch_hessian=8)
```
Note that on large instances, the batch computation could be quite heavy on the
GPU's memory.


## MOI wrapper

ExaPF implements a wrapper to [MathOptInterface](https://github.com/jump-dev/MathOptInterface.jl)
that allows to solve the optimal power flow problem directly in the reduced space
induced by the power flow equations:

```julia
nlp = ExaPF.ReducedSpaceEvaluator("case57.m")
optimizer = Ipopt.Optimizer() # MOI optimizer
# Use LBFGS algorithm, as reduced Hessian is not available by default!
MOI.set(optimizer, MOI.RawParameter("hessian_approximation"), "limited-memory")
MOI.set(optimizer, MOI.RawParameter("tol"), 1e-4)
# Solve reduced space problem
solution = ExaPF.optimize!(optimizer, nlp)
```
