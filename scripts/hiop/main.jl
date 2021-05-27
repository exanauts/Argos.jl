using Printf
using Revise
using ExaPF
using ExaOpt
using SuiteSparse

include("wrapper.jl")


datafile = joinpath(dirname(@__FILE__), "..", "..", "..", "ExaPF.jl", "data", "case9.m")

constraints = Function[
    ExaPF.voltage_magnitude_constraints,
    ExaPF.active_power_constraints,
    ExaPF.reactive_power_constraints,
]

pf_solver = ExaPF.NewtonRaphson(tol=1e-10)
nlp = ExaPF.ReducedSpaceEvaluator(datafile; constraints=constraints, powerflow_solver=pf_solver)
slk = ExaPF.SlackEvaluator(nlp, ExaPF.CPU())
aug = @time ExaPF.AugLagEvaluator(slk, ExaPF.initial(slk); c₀=0.1, scale=true)

function hiop_subproblem(aug)
    pb = build!(aug)
    x = ExaPF.initial(aug)
    return ExaPF.optimize!(pb, aug, x)
end
