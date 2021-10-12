# SIAM PP22

Code to reproduce the numerical results presented in *Batched Second-Order Adjoint Sensitivity for Reduced Space Methods*.

The parallel reduction algorithm is constituted by two blocks:
* `BatchTensorProjection`: implement the code for the batch adjoint and the batch forward-over-reverse projection. That part is implemented in [ExaPF](https://github.com/exanauts/ExaPF.jl/).
* `BatchSparseSolve`: uses `cusolverRF` to compute the second-order adjoints in batch. The reduction is implemented [here](https://github.com/exanauts/ExaPF-Opt/blob/master/src/Evaluators/reduced_evaluator.jl#L467-L511), using [BlockPowerFlow.jl](https://github.com/exanauts/BlockPowerFlow.jl/) for a Julia wrapper to `cusolverRF`.


## Installation
This subdirectory is shipped with its own environment, to reproduce the results more easily.
To install all the dependencies (careful, this could take a while as by default CUDA.jl is downloading its own artifact for CUDA):
```shell
$ julia --project
julia> using Pkg ; Pkg.instantiate()

```

## Batch Hessian
To launch the benchmark presented in the article, you need a NVIDIA GPU compatible with CUDA >= 11.0.
The benchmark runs simply as
```julia
julia> include("benchmark_hessian.jl")
julia> launch_benchmark() # take ~ 20 minutes

```
and all results will be exported as text files in the directory `OUTPUTDIR` (by default `papers/pp2022/results/`).
