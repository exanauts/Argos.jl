```@meta
CurrentModule = Argos
```
```@setup madnlp
using LazyArtifacts
using Argos, ExaPF
using SparseArrays

artifact_toml = joinpath(@__DIR__, "..", "..", "..", "Artifacts.toml")
exadata_hash = artifact_hash("ExaData", artifact_toml)
INSTANCES_DIR = joinpath(artifact_path(exadata_hash), "ExaData")

```

# Full-space OPF

Argos is tightly integrated with the nonlinear interior-point solver
[MadNLP](https://github.com/MadNLP/MadNLP.jl). By default, Argos
relies extensively on MadNLP to solve OPF problems, but
other optimization solvers can be called using the [NLPModels](../man/nlpmodel_wrapper.md)
or the [MOI](../man/moi_wrapper.md) interfaces.


We first detail how to solve the OPF in the full-space.
We start by instantiating a [`FullSpaceEvaluator`](@ref):
```@example madnlp
datafile = joinpath(INSTANCES_DIR, "case9.m")
flp = Argos.FullSpaceEvaluator(datafile)

```

## Instantiating MadNLP
By default MadNLP is using [NLPModels](https://github.com/JuliaSmoothOptimizers/NLPModels.jl/)
to represent the nonlinear model internally. Hence,
one has to convert the `FullSpaceEvaluator` to an `OPFModel`:
```@example madnlp
model = Argos.OPFModel(flp)

```
Then, it remains to build a new `MadNLPSolver` instance
attached to `model`:
```@example madnlp
using MadNLP
solver = MadNLP.MadNLPSolver(model)

```
It just remains to solve the OPF with MadNLP:
```@repl madnlp
stats = MadNLP.solve!(solver)

```

## Querying the solution

MadNLP returns a `MadNLPExecutionStats` object storing the solution.
One can query the optimal objective as:
```@example madnlp
stats.objective
```
and the optimal solution:
```@example madnlp
stats.solution
```

Also, remind that each time the callback [`update!`](@ref)
is being called, the values are updated internally in the
stack stored inside `flp`. Hence, an alternative way to query the
solution is to directly have a look at the values in the stack.
For instance, one can query the optimal values of the voltage
```@example madnlp
stack = flp.stack
[stack.vmag stack.vang]
```
and of the power generation:
```@example madnlp
stack.pgen
```

!!! info
    The values inside `stack` are used to compute the initial point
    in the optimization routine. Hence, if one call `solve!` again
    the optimization would start from the optimal solution found
    in the previous call to `solve!`, leading to a different convergence pattern.
    If one want to launch a new optimization from scratch without reinitializing
    all the data structure, we recommend using the [`reset!`](@ref) function:
    ```@example madnlp
    Argos.reset!(flp)
    ```

## Playing with different parameters

MadNLP has [different options](https://madnlp.github.io/MadNLP.jl/dev/options/) we may want to tune when solving the OPF.
For instance, we can loosen the tolerance to `1e-5`
and set the maximum number of iterations to `5` with:
```@repl madnlp
solver = MadNLP.MadNLPSolver(model; tol=1e-5, max_iter=5)
MadNLP.solve!(solver)
```

Most importantly, one may want to use a different sparse linear
solver than UMFPACK, employed by default in MadNLP.
We recommend using HSL solvers (the installation procedure is
detailed [here](https://madnlp.github.io/MadNLP.jl/dev/installation/#HSL-linear-solver)).
Once HSL installed, one can solve the OPF with:
```julia
using MadNLPHSL
solver = MadNLP.MadNLPSolver(model; linear_solver=Ma27Solver)
MadNLP.solve!(solver)
```

