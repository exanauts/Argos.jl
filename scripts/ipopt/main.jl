

using Printf
using Revise
using ExaPF
using ExaOpt
using Ipopt
using SuiteSparse

using MathOptInterface

const MOI = MathOptInterface

# Remove iterative refinement in UMFPACK
SuiteSparse.UMFPACK.umf_ctrl[8] = 0.0

function subproblem(nlp; ρ=0.1, scale=true)
    x0 = ExaPF.initial(nlp)
    aug = @time ExaPF.AugLagEvaluator(nlp, x0; c₀=ρ, scale=scale)
    solver = Ipopt.Optimizer()
    MOI.set(solver, MOI.RawParameter("print_level"), 5)
    MOI.set(solver, MOI.RawParameter("tol"), 1e-4)
    return ExaPF.optimize!(solver, aug, x0)
end

# Augmented Lagrangian method
function auglag(nlp, x0; iter_max=200, itout_max=1)
    # Init a penalty evaluator with initial penalty c₀
    algo = ExaOpt.AugLagSolver(;
        max_iter=30,
        max_inner_iter=iter_max,
        scaling=true,
        α0=1.0,
        ρ0=1e-1,
        ωtol=1e-3,
        verbose=1,
        inner_algo=:MOI,
        ε_dual=1e-2,
        ε_primal=1e-3,
        lsq_lambda=false,
    )
    c0 = algo.ρ0

    aug = @time ExaPF.AugLagEvaluator(nlp, x0; c₀=c0, scale=true)
    opt = () -> Ipopt.Optimizer(print_level=0, tol=1e-4)
    solution = @time ExaPF.optimize!(algo, aug, x0; moi_optimizer=opt)
    return aug, solution
end

datafile = joinpath(dirname(@__FILE__), "..", "..", "..", "ExaPF.jl", "data", "case9.m")
datafile = joinpath(dirname(@__FILE__), "..", "..", "..", "ExaPF.jl", "data", "case57.m")
datafile = joinpath(dirname(@__FILE__), "..","..", "..", "ExaPF.jl", "data", "case300.m")
# datafile = "../pglib-opf/pglib_opf_case57_ieee.m"
# datafile = "../../pglib-opf/pglib_opf_case118_ieee.m"
# datafile = "../../pglib-opf/pglib_opf_case1354_pegase.m"
# datafile = "../pglib-opf/pglib_opf_case9241_pegase.m"

constraints = Function[
    ExaPF.voltage_magnitude_constraints,
    ExaPF.active_power_constraints,
    ExaPF.reactive_power_constraints,
    # ExaPF.flow_constraints
]

pf_solver = ExaPF.NewtonRaphson(tol=1e-10)
nlp = ExaPF.ReducedSpaceEvaluator(datafile; constraints=constraints, powerflow_solver=pf_solver)
slk = ExaPF.SlackEvaluator(nlp, ExaPF.CPU())
# u0 = fsol.minimizer
# fsol = @time ExaOpt.tron_solve(ExaPF.FeasibilityEvaluator(slk), u0)
# aug, solution = auglag(slk, u0; itout_max=20, iter_max=2000)
sol = subproblem(slk)
