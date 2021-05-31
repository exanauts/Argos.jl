
using Revise
using ExaPF
using MadNLP
using MathOptInterface
using ExaOpt
using SuiteSparse
using CUDA

const MOI = MathOptInterface

if has_cuda_gpu()
    device!(1)
end

SuiteSparse.UMFPACK.umf_ctrl[8] = 0.0

PROBLEMS = Dict(
    # "case9" => "/home/fpacaud/exa/ExaPF.jl/data/case9.m",
    # "case30" => "/home/fpacaud/exa/ExaPF.jl/data/case30.m",
    "case57" => "/home/frapac/dev/anl/ExaPF.jl/data/case57.m",
    "case118" => "/home/fpacaud/exa/ExaPF.jl/data/case118.m",
    "case300" => "/home/fpacaud/exa/ExaPF.jl/data/case300.m",
    "case1354" => "/home/fpacaud/exa/pglib-opf/pglib_opf_case1354_pegase.m",
    "case2869" => "/home/fpacaud/exa/pglib-opf/pglib_opf_case2869_pegase.m",
    "case9241" => "/home/fpacaud/exa/pglib-opf/pglib_opf_case9241_pegase.m",
    # "case13659" => "/home/fpacaud/exa/pglib-opf/pglib_opf_case13659_pegase.m",
    # "case13K" => "/home/fpacaud/exa/ExaPF.jl/data/caseGO13R-025.raw",
)

function build_problem(datafile; scale=true, ρ=0.1, pf_tol=1e-10)
    constraints = [
        ExaPF.voltage_magnitude_constraints,
        ExaPF.active_power_constraints,
        ExaPF.reactive_power_constraints,
    ]
    nlp = ExaPF.ReducedSpaceEvaluator(datafile; powerflow_solver=ExaPF.NewtonRaphson(tol=pf_tol), constraints=constraints)
    slk = ExaPF.SlackEvaluator(nlp, ExaPF.CPU())
    x0 = ExaPF.initial(slk)
    return ExaPF.AugLagEvaluator(slk, x0; c₀=ρ, scale=scale)
end

function madnlp_subproblem(aug)
    optimizer = MadNLP.Optimizer(linear_solver=MadNLP.LapackCPU)
    MOI.set(optimizer, MOI.RawParameter("tol"), 1e-5)
    solution = @time ExaPF.optimize!(optimizer, aug)
    MOI.empty!(optimizer)
end

function solve_auglag(aug)
    algo = ExaOpt.AugLagSolver(;
        max_iter=10,
        max_inner_iter=100,
        scaling=true,
        α0=1.0,
        ρ0=1e1,
        rate=2.0,
        ωtol=1e-5,
        verbose=1,
        inner_algo=:MOI,
        ε_dual=1e-2,
        ε_primal=1e-3,
        # lsq_lambda=true,
    )

    c0 = algo.ρ0

    opt = () -> MadNLP.Optimizer(
        linear_solver=MadNLP.LapackGPU,
        print_level=MadNLP.ERROR,
        max_iter=1000,
        tol=1e-5
    )

    solution = ExaPF.optimize!(algo, aug, x0; moi_optimizer=opt)
    return aug, solution
end

datafile = PROBLEMS["case57"]
aug = build_problem(datafile)
solution = @time madnlp_subproblem(aug)

