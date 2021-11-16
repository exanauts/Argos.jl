using Printf
using Revise
using ExaPF
using Argos
using SuiteSparse

include("wrapper.jl")


datafile = joinpath(dirname(@__FILE__), "..", "..", "..", "ExaPF.jl", "data", "case9.m")

constraints = Function[
    ExaPF.voltage_magnitude_constraints,
    ExaPF.active_power_constraints,
    ExaPF.reactive_power_constraints,
]

pf_solver = ExaPF.NewtonRaphson(tol=1e-10)
nlp = Argos.ReducedSpaceEvaluator(datafile; constraints=constraints, powerflow_solver=pf_solver)
slk = Argos.SlackEvaluator(nlp, ExaPF.CPU())
aug = @time Argos.AugLagEvaluator(slk, ExaPF.initial(slk); c₀=0.1, scale=true)

function hiop_subproblem(aug)
    pb = build!(aug)
    x = Argos.initial(aug)
    return Argos.optimize!(pb, aug, x)
end
