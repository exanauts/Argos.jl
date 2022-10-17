```@meta
CurrentModule = Argos
```
```@setup reduced
using LazyArtifacts
using Argos, ExaPF
using SparseArrays

artifact_toml = joinpath(@__DIR__, "..", "..", "..", "Artifacts.toml")
exadata_hash = artifact_hash("ExaData", artifact_toml)
INSTANCES_DIR = joinpath(artifact_path(exadata_hash), "ExaData")

```

# ReducedSpaceEvaluator

On the contrary to the [`FullSpaceEvaluator`](@ref), the
[`ReducedSpaceEvaluator`](@ref) works in the reduced-space
induced by the power flow equations. Numerically,
this amounts to solve
the system of nonlinear equations $g(x, u) =0$ at each iteration,
in order to find to find a state $x(u)$ satisfying
```math
g(x(u), u) = 0
```
In short, the reduced-space method satisfies automatically
the power flow equations, at the price of solving
a system of nonlinear equations at each iteration.
The OPF problem reformulates in the reduced-space:
```math
\min_{u } \;              f(x(u), u)
\quad \text{subject to}\quad
                         h_l \leq h(x(u), u) \leq h_u .
```
The state $x$ depends *implicitly* from the control $u$, and is
removed from the optimization variables.

The associated reduced gradient and reduced Hessian can be evaluated
using respectively the *adjoint* and the *adjoint-adjoint* methods,
as described [in this paper](https://epubs.siam.org/doi/abs/10.1137/1.9781611977141.6).

!!! info
    The reduced-space method was one of the first
    method to solve the OPF, as described in the
    [seminal article by Dommel & Tinney](https://ieeexplore.ieee.org/abstract/document/4073461).

## Initialization
A `ReducedSpaceEvaluator` can be instantiated both
from a MATPOWER file:
```@example reduced
datafile = joinpath(INSTANCES_DIR, "case9.m")
red = Argos.ReducedSpaceEvaluator(datafile)

```
or equivalently, from a [`ExaPF.PolarForm`](https://exanauts.github.io/ExaPF.jl/stable/lib/formulations/#ExaPF.PolarForm) object:
```@example reduced
polar = ExaPF.PolarForm(datafile)
red = Argos.ReducedSpaceEvaluator(polar)

```
The `ReducedSpaceEvaluator` is somewhat more complicated than
the `FullSpaceEvaluator`, as it has to solve the power flow equations
at each new evaluation. The full signature (with default arguments) is:
```julia
ReducedSpaceEvaluator(
    model;                    # MATPOWER file or ExaPF.PolarForm object
    line_constraints=true,
    powerflow_solver=NewtonRaphson(1e-10),
    linear_solver=nothing,
    nbatch_hessian=1,
)

```

In detail:
- The argument `line_constraints` activates the line flows constraints in the model (by default `true`).
- The option `powerflow_solver` allows to pass a custom [`NewtonRaphson`](https://exanauts.github.io/ExaPF.jl/stable/lib/formulations/#ExaPF.NewtonRaphson) solver with different verbose level or stopping tolerance (by default the tolerance is set to `1e-10`).
- The option `linear_solver` specifies a custom linear solver to use in the Newton-Raphson algorithm and in the computation of the reduced derivatives. If set to `nothing` (default), Argos fallbacks to UMFPACK on the CPU, [CUSOLVERRF](https://github.com/exanauts/CUSOLVERRF.jl) on CUDA GPU. In addition, one can use any linear-solver compatible with [ExaPF](https://exanauts.github.io/ExaPF.jl/stable/lib/linearsolver/), including [iterative linear solvers](https://exanauts.github.io/ExaPF.jl/stable/lib/linearsolver/#Iterative-solvers).
- Finally, the argument `nbatch_hessian` specifies the number of right-hand-side used when solving a linear system `A X = B` in parallel (useful to streamline the evaluation of the reduced derivatives).


## Attributes

Querying the attributes of a `ReducedSpaceEvaluator` works similarly
as with the `FullSpaceEvaluator`.
One can query the original ExaPF model with
```@example reduced
model = Argos.model(red)

```
The initial variable:
```@example reduced
u = Argos.initial(red)
```
The dimensions of the problem:
```@example reduced
(n, m) = Argos.n_variables(red), Argos.n_constraints(red)
```
The bounds on the control `u`
```@example reduced
ulb, uub = Argos.bounds(red, Argos.Variables())
[ulb uub]

```
The bounds on the constraints:
```@example reduced
clb, cub = Argos.bounds(red, Argos.Constraints())
[clb cub]
```

!!! info
    Note that the number of variables is `n=5`, compared to `n=19` for
    the `FullSpaceEvaluator`: the state `x` has been removed from
    the formulation.

### Reduced callback

Let's have a look at the current cache:
```@example reduced
stack = red.stack
[stack.vmag stack.vang]
```
The voltages are those specified in the MATPOWER file `case9.m`.

Now comes the important part. When we call [`update!`](@ref) on
a `FullSpaceEvaluator`, we just copy the values in `x` to the internal
cache `stack`. The behavior is different for `ReducedSpaceEvaluator`:
in addition to copying the values in `u` to the cache `stack`,
we solve the power flow equations to find the associated
implicit state $x(u)$. This is illustrated by the following lines:
```@example reduced
u .= 1.1
Argos.update!(red, u)
[stack.vmag stack.vang]
```
The values stored in `u` has been copied to the corresponding
entries in `stack` (first three entries in `stack.vmag`, corresponding
to the voltage magnitudes at the REF and the PV nodes). Then, the voltage
magnitudes at the PQ nodes and the voltage angles have been updated implicitly
so that the values stored in `stack` satisfy the power flow equations.

!!! note
    Calling [`update!`](@ref) on a `ReducedSpaceEvaluator` calls a power flow
    solver under the hood (which uses the Newton-Raphson algorithm
    [implemented in ExaPF](https://exanauts.github.io/ExaPF.jl/stable/lib/formulations/#ExaPF.nlsolve!)).


The nonlinear solver can be parameterized when instantiating a new
`ReducedSpaceEvaluator`. For instance, if one wants to display the convergence
of the Newton-Raphson algorithm (`verbose=1`) and loosen the tolerance in
the stopping criterion (`tol=1e-6`), one can simply instantiate `red` as
```@example reduced
red = Argos.ReducedSpaceEvaluator(
    datafile;
    powerflow_solver=NewtonRaphson(tol=1e-6, verbose=1)
)
```
By calling `update!` again, we get:

```@repl reduced
Argos.update!(red, u)

```
with a slightly different solution (as we have loosen the tolerance):
```@example reduced
stack = red.stack
[stack.vmag stack.vang]
```

### Reduced gradient
As for `update!`, evaluating the reduced gradient is slightly more involved
in the reduced-space as it involves the solution of one (sparse) linear system
involving the Jacobian of the power flow $\nabla_x g(x, u)$.
Putting aside this consideration, evaluating the gradient simply
amounts to call:
```@example reduced
g = zeros(n)
Argos.gradient!(red, g, u)
g
```

### Reduced Jacobian and Hessian
Similarly, evaluating the reduced Jacobian and the reduced
Hessian both involve resp. the solution of $n_u$ and $2n_u$ linear equations.

Evaluating the (dense) reduced Jacobian translates to
```@example reduced
J = zeros(m, n)
Argos.jacobian!(red, J, u)
J

```
and similarly, for the (dense) reduced Hessian
```@example reduced
H = zeros(n, n)
Argos.hessian!(red, H, u)
H

```

As we will explain later, the computation of the reduced Jacobian and reduced
Hessian can be streamlined on the GPU.


## Deport on CUDA GPU

Instantiating a `ReducedSpaceEvaluator` on a NVIDIA GPU translates to:
```julia
using CUDAKernels # suppose CUDAKernels has been downloaded
red = Argos.ReducedSpaceEvaluator(datafile; device=CUDADevice(), nbatch_hessian=256)
```
The number of batches `nbatch_hessian` is the number of right-hand-side used
to streamline the solution of the linear systems.
