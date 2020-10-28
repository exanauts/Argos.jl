
using ExaPF
using ExaOpt
using UnicodePlots

function load_nlp(datafile)
    # Load problem.
    pf = ExaPF.PowerSystem.PowerNetwork(datafile, 1)
    polar = PolarForm(pf, ExaPF.CPU())

    x0 = ExaPF.initial(polar, State())
    p = ExaPF.initial(polar, Parameters())
    uk = ExaPF.initial(polar, Control())

    constraints = Function[ExaPF.state_constraint, ExaPF.power_constraints]
    nlp = ExaPF.ReducedSpaceEvaluator(polar, x0, uk, p; constraints=constraints,
                                      ε_tol=1e-10)
    return nlp, uk
end

# Augmented Lagrangian method
function auglag(nlp, u0; iter_max=200, itout_max=1)
    # Init a penalty evaluator with initial penalty c₀
    algo = ExaOpt.AugLagSolver(; max_iter=itout_max)
    c0 = 0.1
    aug = ExaPF.AugLagEvaluator(nlp, u0; c₀=c0, scale=true)
    uk, tracer = ExaOpt.optimize(algo, aug, u0)
    return uk, tracer
end

datafile = joinpath(dirname(@__FILE__), "..", "..", "ExaPF.jl", "test", "data", "case57.m")
# datafile = "../pglib-opf/pglib_opf_case57_ieee.m"
# datafile = "../pglib-opf/pglib_opf_case118_ieee.m"
# datafile = "../pglib-opf/pglib_opf_case1354_pegase.m"
# datafile = "../pglib-opf/pglib_opf_case9241_pegase.m"

nlp, u0 = load_nlp(datafile)
u_opt, tracer= @time auglag(nlp, u0; itout_max=10, iter_max=1000)

