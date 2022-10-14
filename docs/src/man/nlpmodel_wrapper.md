# NLPModels
```@setup nlpmodel
using LazyArtifacts

artifact_toml = joinpath(@__DIR__, "..", "..", "..", "Artifacts.toml")
exadata_hash = artifact_hash("ExaData", artifact_toml)
INSTANCES_DIR = joinpath(artifact_path(exadata_hash), "ExaData")

```

Knowledgeable users may have noticed that the [`Argos.AbstractNLPEvaluator`](@ref)
API is closed to [NLPModels](https://github.com/JuliaSmoothOptimizers/NLPModels.jl/). Hence, it is straightforward
to wrap any `AbstractNLPEvaluator` in a `NLPModels.AbstractNLPModel` structure.

## Initializing
In Argos, this is provided by the `OPFModel` structure, which
takes as input any `AbstractNLPEvaluator` and converts it as a
`NLPModels.AbstractNLPModel`.

```@example nlpmodel
using NLPModels
using Argos

# Import OPF model in Argos
datafile = joinpath(INSTANCES_DIR, "case57.m")
flp = Argos.FullSpaceEvaluator(datafile)

# Convert it to a AbstractNLPModel:
model = Argos.OPFModel(flp)
@assert isa(model, NLPModels.AbstractNLPModel)

```

## Playing with NLPModels' API
The user can use the standard NLPModels API
to interact with the OPF model:

- Querying the number of variables:
```@repl nlpmodel
n = NLPModels.get_nvar(model)
```
- Querying the initial variable `x0`:
```@repl nlpmodel
x0 = NLPModels.get_x0(model)
```
- Evaluating the objective
```@repl nlpmodel
NLPModels.obj(model, x0)
```
- Evaluating the constraints
```@repl nlpmodel
NLPModels.cons(model, x0)
```
- Evaluating the gradient
```@repl nlpmodel
NLPModels.grad(model, x0)
```
and so on...
