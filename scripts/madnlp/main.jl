
using Revise
using ExaPF
using MadNLP
using MadNLPGPU
using MathOptInterface
using ExaOpt
using CUDA

const MOI = MathOptInterface

# MadNLP wrapper
include("wrapper.jl")
include(joinpath(dirname(@__FILE__), "..", "common.jl"))
include(joinpath(dirname(@__FILE__), "..", "problems.jl"))

function madnlp_optimizer(linear_solver)
    opt = MadNLP.Optimizer()
    MOI.set(opt, MOI.RawParameter("linear_solver"), linear_solver)
    MOI.set(opt, MOI.RawParameter("print_level"), MadNLP.ERROR)
    MOI.set(opt, MOI.RawParameter("max_iter"), 1000)
    MOI.set(opt, MOI.RawParameter("tol"), 1e-5)
    return opt
end

function madnlp_subproblem(aug; linear_solver=MadNLPLapackCPU)
    optimizer = MadNLP.Optimizer(linear_solver=linear_solver)
    MOI.set(optimizer, MOI.RawParameter("tol"), 1e-5)
    solution = @time ExaOpt.optimize!(optimizer, aug)
    MOI.empty!(optimizer)
end

function solve_auglag(aug; linear_solver=MadNLPLapackCPU, max_iter=20, penalty=0.1, rate=10.0)
    algo = ExaOpt.AugLagSolver(;
        max_iter=max_iter,
        max_inner_iter=100,
        α0=1.0,
        ρ0=penalty,
        rate=rate,
        ωtol=1e-5,
        verbose=1,
        inner_algo=:MOI,
        ε_dual=1e-2,
        ε_primal=1e-3,
    )

    x0 = ExaOpt.initial(aug)
    aug.ρ = penalty # update penalty in Evaluator

    optimizer = () -> madnlp_optimizer(linear_solver)

    solution = ExaOpt.optimize!(algo, aug, x0; moi_optimizer=optimizer)
    return aug, solution
end

datafile = PROBLEMS["case300"]
aug = build_problem(datafile)
solution = @time solve_auglag(aug)

