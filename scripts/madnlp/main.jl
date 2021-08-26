
using Revise
using ExaPF
using MadNLP
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
    ExaOpt.reset!(aug)
    optimizer = MadNLP.Optimizer(linear_solver=linear_solver)
    MOI.set(optimizer, MOI.RawParameter("tol"), 1e-5)
    solution = @time ExaOpt.optimize!(optimizer, aug)
    MOI.empty!(optimizer)
    return solution
end

function solve_auglag_moi(aug; linear_solver=MadNLPLapackCPU, max_iter=20, penalty=0.1, rate=10.0)
    options = ExaOpt.AugLagOptions(;
        max_iter=max_iter,
        max_inner_iter=100,
        α0=1.0,
        rate=rate,
        ωtol=1e-5,
        verbose=1,
        ε_dual=1e-2,
        ε_primal=1e-5,
    )
    optimizer = MadNLP.Optimizer(
        linear_solver=linear_solver, print_level=MadNLP.ERROR, max_iter=300,
    )
    solver = ExaOpt.AuglagSolver(optimizer, options)

    x0 = ExaOpt.initial(aug)
    aug.ρ = penalty # update penalty in Evaluator

    return ExaOpt.optimize!(solver, aug, x0)
end

datafile = PROBLEMS["case300"]
aug = build_problem(datafile)

# solution = @time solve_auglag(aug, max_iter=10, penalty=1.0)

