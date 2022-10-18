```@meta
CurrentModule = Argos
```
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
In Argos, this is provided by the [`OPFModel`](@ref) structure, which
takes as input any `AbstractNLPEvaluator` and converts it as an `NLPModels``.`AbstractNLPModel`.

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


## Accelerating the callbacks on an NVIDIA GPU

We can exploit any available NVIDIA GPU to accelerate the evaluation
of the derivatives. To do so, one first needs to
[install](../quickstart/cuda.md) `ArgosCUDA`](../quickstart/cuda.md).

Then, we can instantiate a new evaluator on the GPU with:
```julia
using ArgosCUDA, CUDAKernels
flp = Argos.FullSpaceEvaluator(datafile; device=CUDADevice())

```
The [`OPFModel`](@ref) structure works exclusively on the host memory,
so we have to bridge the evaluator `flp` to the host before creating
a new instance of `OPFModel`:
```julia
brige = Argos.bridge(flp)
model = Argos.OPFModel(bridge)

```

!!! note
    Bridging an evaluator between the host and the device induces
    significant data movements between the host and the device, as for each
    input or for each output we have to move the data back and forth between
    the host and the device. However, we have noticed that in practice
    the time to operate the data transfer is negligible compared to the other
    operations (linear algebra, KKT system solution) pursued inside the
    optimization algorithm.

