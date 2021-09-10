
using Revise, MadNLP, ExaOpt
using MadNLPGPU

# Load GPU interface
include(joinpath(@__DIR__, "..", "madnlp", "main.jl"))
include(joinpath(@__DIR__, "..", "madnlp", "wrapper.jl"))
include(joinpath(@__DIR__, "..", "..", "test", "cusolver.jl"))
# Instance
datafile = PROBLEMS["case1354"]

INSTANTIATE = true
MAX_BATCHES = 250
SCALE = true
CPU_SOLVE = true


if INSTANTIATE && CPU_SOLVE
    @info "Instantiate problem on CPU"
    aug_cpu = @time build_problem(datafile)
    nlp_cpu = ExaOpt.inner_evaluator(aug_cpu)
end

if INSTANTIATE
    # Params
    nbatches = min(MAX_BATCHES, ExaOpt.n_variables(nlp_cpu))
    linear_solver = MadNLPLapackGPU

    # GPU evaluators
    # Wrap GPU callbacks to pass Hessian matrix on the host
    @info "Instantiate problem on GPU (wrapped)"
    aug_wrapped = @time build_batch_problem(datafile, nbatches; scale=SCALE, wrap=true)
    # Wrap GPU callbacks to pass Hessian matrix on the host
    @info "Instantiate problem on GPU (unwrapped)"
    aug_unwrapped = @time build_batch_problem(datafile, nbatches; scale=SCALE, wrap=false)
end

if CPU_SOLVE
    # Test resolution with MOI (KKT: sparse COO)
    @info "Resolution with MOI + Lapack CPU"
    madnlp_subproblem(aug_cpu)
end

# Test resolution with dense interface (KKT: dense matrix)
@info "Resolution with dense MadNLP"
test_dense(aug_wrapped; linear_solver=linear_solver)

# Test resolution with dense interface on GPU
@info "Resolution with dense MadNLP + GPU"
test_dense_gpu(aug_unwrapped)

# Test resolution with dense interface on GPU + Schur resolution for linear system
@info "Resolution with dense MadNLP + GPU + Schur "
test_dense_gpu_new(aug_unwrapped)

# Clean
GC.gc(true)
CUDA.reclaim()
