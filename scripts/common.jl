
using ExaPF
using LinearAlgebra
using KernelAbstractions
using CUDA
using CUDA.CUSPARSE
using BlockPowerFlow

using CUDAKernels

function build_batch_auglag(datafile, device, nbatch; ρ=0.1, scale=false)
    pf_solver = NewtonRaphson(; tol=1e-10)
    nlp = ExaOpt.ReducedSpaceEvaluator(
        datafile; device=device, nbatch_hessian=nbatch, powerflow_solver=pf_solver,
    )
    bdg = ExaOpt.BridgeDeviceEvaluator(nlp, CuArray{Float64, 1}, CuArray{Float64, 2})
    slk = ExaOpt.SlackEvaluator(bdg)
    x0 = ExaOpt.initial(slk)
    aug = ExaOpt.AugLagEvaluator(slk, x0; c₀=ρ, scale=scale)
    return aug
end

