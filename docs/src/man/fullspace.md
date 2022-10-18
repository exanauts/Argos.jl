```@meta
CurrentModule = Argos
```
```@setup full_evaluator
using LazyArtifacts
using Argos, ExaPF
using SparseArrays

artifact_toml = joinpath(@__DIR__, "..", "..", "..", "Artifacts.toml")
exadata_hash = artifact_hash("ExaData", artifact_toml)
INSTANCES_DIR = joinpath(artifact_path(exadata_hash), "ExaData")

```
# FullSpaceEvaluator

[`FullSpaceEvaluator`](@ref) models the original OPF
problem in the full-space.

## Initialization
A `FullSpaceEvaluator` can be instantiated both
from a MATPOWER file:
```@example full_evaluator
datafile = joinpath(INSTANCES_DIR, "case9.m")
flp = Argos.FullSpaceEvaluator(datafile)

```
or equivalently, from a [`ExaPF.PolarForm`](https://exanauts.github.io/ExaPF.jl/stable/lib/formulations/#ExaPF.PolarForm) object:
```@example full_evaluator
polar = ExaPF.PolarForm(datafile)
flp = Argos.FullSpaceEvaluator(polar)

```

!!! note
    One can remove the line-flow constraints in the model
    simply by adding the keyword argument `flp = Argos.FullSpaceEvaluator(polar; line_constraints=false)`.

## Attributes

[`FullSpaceEvaluator`](@ref) is just a thin wrapper on top of ExaPF,
and comes with just a few attributes. One can query the original
ExaPF model with
```@example full_evaluator
model = Argos.model(flp)

```
The initial variable:
```@example full_evaluator
x = Argos.initial(flp)
```
The dimensions of the problem:
```@example full_evaluator
(n, m) = Argos.n_variables(flp), Argos.n_constraints(flp)
```
The bounds on the variables `(x, u)`
```@example full_evaluator
xlb, xub = Argos.bounds(flp, Argos.Variables())
[xlb xub]

```
The bounds on the constraints:
```@example full_evaluator
clb, cub = Argos.bounds(flp, Argos.Constraints())
[clb cub]
```

### Sparse Jacobian and sparse Hessian
More importantly, `FullSpaceEvaluator` stores
two [AD backends](https://exanauts.github.io/ExaPF.jl/stable/man/autodiff/) to evaluate the Jacobian and the
Hessian in sparse format using ExaPF.

For the Hessian, the AD backend is
```@example full_evaluator
flp.hess

```
generating the matrix
```@example full_evaluator
flp.hess.H

```
and for the Jacobian
```@example full_evaluator
flp.jac

```
generating the matrix
```@example full_evaluator
flp.jac.J

```
Both AD backends use coloring to reduce the number
of Hessian-vector and Jacobian-vector products required
to evaluate the Hessian and Jacobian in sparse format.

To avoid dealing explicitly with the AD backends, Argos
provides a function to query directly the Hessian and Jacobian
in COO format:
```@example full_evaluator
# Query sparsity pattern:
j_I, j_J = Argos.jacobian_structure(flp)
nnzj = length(j_I)
j_V = zeros(nnzj)
Argos.jacobian_coo!(flp, j_V, x)
sparse(j_I, j_J, j_V) # build a SparseMatrixCSC
```
and for the Hessian:
```@example full_evaluator
# Query sparsity pattern:
h_I, h_J = Argos.hessian_structure(flp)
nnzh = length(h_I)
h_V = zeros(nnzh)
y = rand(m)
Argos.hessian_lagrangian_coo!(flp, h_V, x, y, 1.0)
sparse(h_I, h_J, h_V) # build a SparseMatrixCSC
```

!!! info
    For the Hessian, only the lower-triangular are being returned.

## Deport on CUDA GPU
Deporting all the operations on a CUDA GPU simply amounts to instantiating a
`FullSpaceEvaluator``](@ref) on the GPU, with
```julia
using CUDAKernels # suppose CUDAKernels has been downloaded
flp = Argos.FullSpaceEvaluator(datafile; device=CUDADevice())
```

Then, the API remains exactly the same as on the CPU.

When using `device=CUDADevice()`, the model is entirely instantiated on the device,
without data left on the host (hence minimizing the communication
costs). The computation of the derivatives is streamlined by propagating
the tangents in parallel, leading to faster evaluations of the callbacks.
As expected, the larger the model, the more significant the performance gain.

