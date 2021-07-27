

using Printf
using Revise
using ExaPF
using ExaOpt
using Ipopt
using SuiteSparse

using MathOptInterface
const MOI = MathOptInterface

include(joinpath(dirname(@__FILE__), "..", "common.jl"))
include(joinpath(dirname(@__FILE__), "..", "problems.jl"))

function ipopt_optimizer()
    solver = Ipopt.Optimizer()
    # Default parameters
    MOI.set(solver, MOI.RawParameter("print_level"), 0)
    MOI.set(solver, MOI.RawParameter("tol"), 1e-4)
    return solver
end

function ipopt_subproblem(aug)
    optimizer = ipopt_optimizer()
    MOI.set(solver, MOI.RawParameter("print_level"), 5)
    solution = @time ExaOpt.optimize!(optimizer, aug)
    MOI.empty!(optimizer)
end

# Augmented Lagrangian method
function solve_auglag(aug; max_iter=20, penalty=0.1, rate=10.0)
    # Init a penalty evaluator with initial penalty c₀
    algo = ExaOpt.AugLagSolver(;
        max_iter=max_iter,
        max_inner_iter=100,
        α0=1.0,
        ρ0=penalty,
        rate=rate,
        ωtol=1e-3,
        verbose=1,
        inner_algo=:MOI,
        ε_dual=1e-2,
        ε_primal=1e-3,
    )
    aug.ρ = penalty
    x0 = ExaOpt.initial(aug)

    solution = @time ExaOpt.optimize!(algo, aug, x0; moi_optimizer=ipopt_optimizer)
    return aug, solution
end

datafile = PROBLEMS["case300"]
aug = build_problem(datafile)
solution = @time solve_auglag(aug)

