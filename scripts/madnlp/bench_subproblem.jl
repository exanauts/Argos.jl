
using Revise, MadNLP, Argos
using MadNLPGPU

# Load GPU interface
include(joinpath(@__DIR__, "..", "madnlp", "main.jl"))
include(joinpath(@__DIR__, "..", "..", "test", "cusolver.jl"))
# Instance
datafile = joinpath(dirname(pathof(Argos)), "..", "data", "case1354pegase.m")

INSTANTIATE = true
MAX_BATCHES = 250
SCALE = true
CPU_SOLVE = true


if INSTANTIATE && CPU_SOLVE
    @info "Instantiate problem on CPU"
    aug_cpu = @time Argos.instantiate_auglag_model(datafile)
    nlp_cpu = Argos.inner_evaluator(aug_cpu)
end

if INSTANTIATE
    # Params
    nbatches = min(MAX_BATCHES, Argos.n_variables(nlp_cpu))
    linear_solver = MadNLPLapackGPU

    # GPU evaluators
    # Wrap GPU callbacks to pass Hessian matrix on the host
    @info "Instantiate problem on GPU (wrapped)"
    aug_wrapped = @time Argos.instantiate_auglag_model(datafile; nbatches=nbatches, scale=SCALE, wrap=true, device=CUDADevice())
    # Wrap GPU callbacks to pass Hessian matrix on the host
    @info "Instantiate problem on GPU (unwrapped)"
    aug_unwrapped = @time Argos.instantiate_auglag_model(datafile; nbatches=nbatches, scale=SCALE, wrap=false, device=CUDADevice())
end

if CPU_SOLVE
    # Test resolution with MOI (KKT: sparse COO)
    @info "Resolution with MOI + Lapack CPU"
    subproblem_moi(aug_cpu)
end

# Test resolution with dense interface (KKT: dense matrix)
@info "Resolution with dense MadNLP"
subproblem_dense_kkt(aug_wrapped; linear_solver=linear_solver)

# Test resolution with dense interface on GPU
@info "Resolution with dense MadNLP + GPU"
subproblem_dense_kkt_gpu(aug_unwrapped)

# Test resolution with dense interface on GPU + Schur resolution for linear system
@info "Resolution with dense MadNLP + GPU + Schur "
subproblem_schur_kkt_gpu(aug_unwrapped)

# Clean
GC.gc(true)
CUDA.reclaim()
