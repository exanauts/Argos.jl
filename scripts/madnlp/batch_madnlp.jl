
using Revise
using ExaPF
using MadNLP
using MathOptInterface
using ExaOpt
using SuiteSparse
using CUDA
using CUDA.CUSPARSE
using BlockPowerFlow

using KernelAbstractions
using CUDAKernels

const MOI = MathOptInterface
const LS = ExaPF.LinearSolvers
const PS = ExaPF.PowerSystem

if has_cuda_gpu()
    device!(1)
end

SuiteSparse.UMFPACK.umf_ctrl[8] = 0.0

include(joinpath(dirname(@__FILE__), "..", "common.jl"))

function batch_auglag(datafile, nbatch)
    algo = ExaOpt.AugLagSolver(;
        max_iter=1,
        max_inner_iter=100,
        scaling=true,
        α0=1.0,
        ρ0=10.0,
        rate=2.0,
        ωtol=1e-5,
        verbose=1,
        inner_algo=:MOI,
        ε_dual=1e-2,
        ε_primal=1e-3,
        # lsq_lambda=true,
    )

    c0 = algo.ρ0

    constraints = [
        ExaPF.voltage_magnitude_constraints,
        ExaPF.active_power_constraints,
        ExaPF.reactive_power_constraints,
    ]

    nlp = build_batch_nlp(datafile, CUDADevice(), nbatch)
    slk = ExaPF.SlackEvaluator(nlp, ExaPF.CPU())
    x0 = ExaPF.initial(slk)
    aug = ExaPF.AugLagEvaluator(slk, x0; c₀=c0, scale=false)
    opt = () -> MadNLP.Optimizer(
        linear_solver=MadNLP.LapackGPU,
        print_level=MadNLP.INFO,
        max_iter=5,
        tol=1e-5
    )
    solution = @profile ExaPF.optimize!(algo, aug, x0; moi_optimizer=opt)
    return aug, solution
end

datafile = PROBLEMS["case30"]
aug, solution = @time batch_auglag(datafile, 1)
