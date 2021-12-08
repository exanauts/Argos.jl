# SIAM PP22

Code to reproduce the numerical results presented in *Batched Second-Order Adjoint Sensitivity for Reduced Space Methods*.

The parallel reduction algorithm is constituted by two blocks:
* `BatchTensorProjection`: implement the code for the batch adjoint and the batch forward-over-reverse projection. That part is implemented in [ExaPF](https://github.com/exanauts/ExaPF.jl/).
* `BatchSparseSolve`: uses `cusolverRF` to compute the second-order adjoints in batch. The reduction is implemented [here](https://github.com/exanauts/Argos.jl/blob/master/src/Evaluators/reduced_evaluator.jl#L467-L511), using [BlockPowerFlow.jl](https://github.com/exanauts/BlockPowerFlow.jl/) for a Julia wrapper to `cusolverRF`.

The GPU used throughout the numerical experiments is a NVIDIA V100 GPU.
The code has been tested on Julia 1.6 and CUDA 11.3.0, and the following packages:
```julia
(pp2022) pkg> st
      Status `~/exa/Argos/papers/pp2022/Project.toml`
  [0ca6dcf5] BlockPowerFlow v0.1.0 `https://github.com/exanauts/BlockPowerFlow.jl.git#master`
  [052768ef] CUDA v3.3.6
  [72cfdca4] CUDAKernels v0.3.0
  [ef244971] ExaOpt v0.1.0 `https://github.com/exanauts/ExaPF-Opt.git#pp2022`
  [0cf0e50c] ExaPF v0.5.0 `https://github.com/exanauts/ExaPF.jl.git#fp/exaopt`
  [63c18a36] KernelAbstractions v0.7.0
```
See the instructions below to install a fresh environment to reproduce the results.


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
If the directory does not exist, it is created automatically inside the script (require permission to create the directory locally).
