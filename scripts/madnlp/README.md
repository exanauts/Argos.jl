# MadNLP wrapper

This directory stores a few scripts to test and benchmark the MadNLP wrapper.

## Wrapper
The wrapper is implemented in `src/Wrappers/MadNLP_wrapper.jl`, and is shipped in two parts:
1. First, it brings a thin wrapper to bridge `ExaOpt.AbstractNLPEvaluator` to `NLPModels.AbstractNLPModel` (remind that the `AbstractNLPEvaluator` abstraction was inspired by `NLPModels`)
2. Second, it implements a custom `MadNLP.AbstractKKTSystem` to exploit the structure of Augmented Lagrangian's subproblems: `ExaOpt.MixedAuglagKKTSystem`.

The wrapper is directly exploited in the Augmented Lagrangian algorithm implemented in `src/Algorithms/auglag.jl`.
MadNLP is able to solve the Auglag subproblems efficiently, using the function [`solve_subproblem`](https://github.com/exanauts/ExaPF-Opt/blob/master/src/Algorithms/auglag.jl#L45-L70). The initial barrier parameter `mu` is tuned to warm-start efficiently the solver.

## Usage

MadNLP can be used both on the CPU and on the GPU. The following subsections
detail the code implemented in `scripts/madnlp/main.jl`.

Before all, you need to load ExaOpt's extension to use the GPU:
```julia
using MadNLPGPU # allow to use MadNLP on the GPU
include("test/cusolver.jl") # allow to use cusolver with ExaOpt

```

The next two paragraphs detail line by line the code called in the script
`scripts/madnlp/bench_subproblem.jl`, benchmarking the different KKTSystem
we can use in MadNLP with ExaOpt.


### CPU
First, you need to instantiate a new `AugLagEvaluator` object:
```julia
datafile = "data/case300.m" # any matpower instances
aug_cpu = ExaOpt.instantiate_auglag_model(datafile; line_constraints=false)

```
You can set the penalty parameter in `AugLagEvaluator` with:
```julia
aug_cpu.ρ = 10.0 # 10.0 by default
```
The larger the penalty, the worst is the conditioning of the KKT system.

By default, MadNLP uses MOI to solve the Auglag's subproblem:
```julia
ExaOpt.reset!(aug_cpu) # reset data inside aug_cpu
optimizer = MadNLP.Optimizer(linear_solver=linear_solver, tol=1e-5)
# Solve subproblem with MOI
solution = @time ExaOpt.optimize!(optimizer, aug_cpu)

```

In that case, the problem is handled in sparse format inside MadNLP.
Instead, it's more efficient to exploit the structure directly in dense format,
with a `DenseKKTSystem`:
```julia
ExaOpt.reset!(aug_cpu) # reset data inside aug_cpu
mnlp = ExaOpt.ExaNLPModel(aug_cpu) # instantiate a NLPModels wrapper aug_cpu
options = Dict{Symbol, Any}(
    :tol=>1e-5,
    :kkt_system=>MadNLP.DENSE_KKT_SYSTEM,
    :linear_solver=>MadNLPLapackCPU,
)
ipp = MadNLP.InteriorPointSolver(mnlp; option_dict=options)
solution = @time MadNLP.optimize!(ipp)

```
The code presented in the last block solves the optimization
problem with i) a `DenseKKTSystem` abstraction (specified with the option `kkt_system`)
ii) the linear solver Lapack (option `linear_solver`).

Using a `MixedAuglagKKTSystem` is more complicated (the current API in MadNLP is not great,
but should be improved soon). The procedure is:
```julia
mnlp = ExaOpt.ExaNLPModel(aug_cpu) # instantiate a NLPModels wrapper aug_cpu
madopt = MadNLP.Options(linear_solver=MadNLPLapackCPU)
options = Dict{Symbol, Any}(
    :tol=>1e-5,
    :linear_solver=>MadNLPLapackCPU,
)
MadNLP.set_options!(madopt,options,Dict())
# Type for Custom KKT system
KKT = ExaOpt.MixedAuglagKKTSystem{Float64, Vector{Float64}, Matrix{Float64}}
ipp = MadNLP.InteriorPointSolver{KKT}(mnlp, madopt; option_linear_solver=options)
solution = @time MadNLP.optimize!(ipp)

```

### CUDA GPU
The procedure looks exactly the same to solve the problems on CUDA GPU.
First, we need to instantiate a new evaluator on the GPU:
```julia
nbatches = 250 # number of batches used to compute the reduced Hessian in parallel
aug_gpu = ExaOpt.instantiate_auglag_model(datafile; line_constraints=false, device=CUDADevice(), nbatches=nbatches)
```
You could verify that `aug_gpu` is well instantiate by calling:
```julia
x0 = ExaOpt.initial(aug_gpu)
```
Normally, `x0` is a `CuArray`.

Then, we can repeat the same procedure as previously (the MOI wrapper don't work when `aug_gpu` is instantiated
on the GPU).

To solve the Auglag's subproblem with a `DenseKKTSystem`:
```julia
ExaOpt.reset!(aug_gpu) # reset evaluator
mnlp = ExaOpt.ExaNLPModel(aug_gpu)
options = Dict{Symbol, Any}(:tol=>1e-5,
                            :linear_solver=>MadNLPLapackGPU)
madopt = MadNLP.Options(linear_solver=linear_solver)
MadNLP.set_options!(madopt,options,Dict())
# Allocate DenseKKTSystem on the GPU
KKT = MadNLP.DenseKKTSystem{Float64, CuVector{Float64}, CuMatrix{Float64}}
ipp = MadNLP.InteriorPointSolver{KKT}(mnlp, madopt; option_linear_solver=options)
@time MadNLP.optimize!(ipp)

```

The same procedure holds to instantiate a `MixedAuglagKKTSystem` on the GPU:
```julia
ExaOpt.reset!(aug_gpu)
mnlp = ExaOpt.ExaNLPModel(aug_gpu)
options = Dict{Symbol, Any}(:tol=>1e-5,
                            :linear_solver=>MadNLPLapackGPU)
madopt = MadNLP.Options(linear_solver=linear_solver)
MadNLP.set_options!(madopt,options,Dict())
# Custom KKT type
KKT = ExaOpt.MixedAuglagKKTSystem{Float64, CuVector{Float64}, CuMatrix{Float64}}
ipp = MadNLP.InteriorPointSolver{KKT}(mnlp, madopt; option_linear_solver=options)
@time MadNLP.optimize!(ipp)

```
