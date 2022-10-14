```@meta
CurrentModule = Argos
```
```@setup evaluator
using LazyArtifacts

artifact_toml = joinpath(@__DIR__, "..", "..", "..", "Artifacts.toml")
exadata_hash = artifact_hash("ExaData", artifact_toml)
INSTANCES_DIR = joinpath(artifact_path(exadata_hash), "ExaData")

```

# Overview

Argos acts as a layer build on top of [ExaPF](https://github.com/exanauts/ExaPF.jl/)
to formulate the OPF problems in a form suitable
for optimization solvers (i.e. by formulating the problems
with nonlinear callbacks). Depending whether we are working in the full-
or in the reduced-space, the two principal evaluators are
the [`FullSpaceEvaluator`](@ref) and the [`ReducedSpaceEvaluator`](@ref).

## Abstraction

Argos formulates the OPF problem in abstract form as
```math
\min_{x, u } \;              f(x, u)
\quad \text{subject to}\quad
\left\{
\begin{aligned}
 & g(x, u) = 0 \\
                        & h_l \leq h(x, u) \leq h_u .
\end{aligned}
\right.
```
By design, the control variable $u$ (voltage magnitude at PV and REF nodes, active power generations) is dissociated from the state variable $x$ (voltage angle, voltage magnitudes at PQ nodes).
The function $f$ encodes the [objective function](https://exanauts.github.io/ExaPF.jl/stable/lib/formulations/#Objective), the function $g$ is associated
to the [power flow equations](https://exanauts.github.io/ExaPF.jl/stable/lib/formulations/#ExaPF.PowerFlowBalance)
and $h$ encodes the remaining operational constraints
([bounds on reactive power generations](https://exanauts.github.io/ExaPF.jl/stable/lib/formulations/#ExaPF.PowerGenerationBounds),
[line flow constraints](https://exanauts.github.io/ExaPF.jl/stable/lib/formulations/#ExaPF.LineFlows`)).


!!! note
    Compared to MATPOWER or PowerModels, Argos comes with two major
    differences.
    1. Only a subset of the power flow equations is considered, leading to a total of $n_x$ equations instead of $2 n_{bus}$.
    2. The reactive power generations are considered only implicitly, through the remaining power flow equations.

    Hence, the problem formulated by Argos has a smaller size as
    the ones obtained through PowerModels or MATPOWER, at the expense
    of robustness.


## Usage
For demonstration purpose, we instantiate a new `FullSpaceEvaluator` object
associated to `case9`:
```@example evaluator
using ExaPF, Argos
datafile = joinpath(INSTANCES_DIR, "case9.m")
flp = Argos.FullSpaceEvaluator(datafile)

```
We get the number of variables and constraints simply as
```@example evaluator
(n, m) = Argos.n_variables(flp), Argos.n_constraints(flp)

```

One can

### Network variables
Internally, each evaluator stores the current state of the network
in a [`ExaPF.NetworkStack`](https://exanauts.github.io/ExaPF.jl/stable/lib/formulations/#ExaPF.NetworkStack) object.
We can query the current state of the network as:
```@example evaluator
stack = flp.stack
# Query current voltage magnitude and angle values
[stack.vmag stack.vang]
```

One updates the values in the cache by calling the function
`update!`:
```@example evaluator
x = rand(n)
Argos.update!(flp, x)
# The values in the cache are modified accordingly
[stack.vmag stack.vang]

```

!!! note
    Everytime we have a new variable `x`, it is important
    to refresh the cache by calling explicitly `Argos.update!(flp, x)`.


### Callbacks

Now the cache has been refreshed by calling `update!`, one can query the
different callbacks:

**Objective:**
```@example evaluator
obj = Argos.objective(flp, x)
```
**Gradient:**
```@example evaluator
g = zeros(n)
Argos.gradient!(flp, g, x)
g
```
**Constraints:**
```@example evaluator
cons = zeros(m)
Argos.constraint!(flp, cons, x)
cons
```

!!! note
    All the callbacks are written to modify the data (constraints, gradient)
    inplace, to avoid unneeded allocations. In addition Argos provides
    a version allocating automatically the return values:
    ```@example evaluator
    g = Argos.gradient(flp, x)
    c = Argos.constraint(flp, x)
    ```

Eventually, one can reset the evaluator to its original
state by using [`reset!`](@ref):
```@example evaluator
Argos.reset!(flp)
[stack.vmag stack.vang]
```
