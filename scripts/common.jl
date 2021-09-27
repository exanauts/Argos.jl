
using ExaPF
using LinearAlgebra
using SuiteSparse
using KernelAbstractions
using CUDA
using BlockPowerFlow

const CUDA_DEVICE = 1
# Remove iterative refinement in UMFPACK
SuiteSparse.UMFPACK.umf_ctrl[8] = 0.0

function build_problem(datafile; scale=true, ρ=10.0, pf_tol=1e-10, line_constraints=false)
    nlp = ExaOpt.ReducedSpaceEvaluator(
        datafile;
        powerflow_solver=ExaPF.NewtonRaphson(tol=pf_tol),
        line_constraints=line_constraints,
    )
    slk = ExaOpt.SlackEvaluator(nlp)
    x0 = ExaOpt.initial(slk)
    return ExaOpt.AugLagEvaluator(slk, x0; c₀=ρ, scale=scale)
end

if has_cuda_gpu()
    using CUDAKernels
    using CUDA.CUSPARSE
    device!(CUDA_DEVICE)

    function build_batch_problem(datafile, nbatch; ρ=10.0, scale=false, wrap=true, line_constraints=false)
        pf_solver = NewtonRaphson(; tol=1e-10)
        nlp = ExaOpt.ReducedSpaceEvaluator(
            datafile; device=CUDADevice(), nbatch_hessian=nbatch, powerflow_solver=pf_solver,
            line_constraints=line_constraints,
        )
        if wrap
            bdg = ExaOpt.BridgeDeviceEvaluator(nlp, CuArray{Float64, 1}, CuArray{Float64, 2})
            slk = ExaOpt.SlackEvaluator(bdg)
        else
            slk = ExaOpt.SlackEvaluator(nlp)
        end
        x0 = ExaOpt.initial(slk)
        aug = ExaOpt.AugLagEvaluator(slk, x0; c₀=ρ, scale=scale)
        return aug
    end
end

