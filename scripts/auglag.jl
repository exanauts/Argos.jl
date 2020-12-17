
using ExaPF
using ExaOpt
using UnicodePlots

# Augmented Lagrangian method
function auglag(nlp, u0; iter_max=200, itout_max=1)
    # Init a penalty evaluator with initial penalty c₀
    algo = ExaOpt.AugLagSolver(; max_iter=itout_max, ωtol=1e-4, verbose=1, inner_algo=:projectedgradient)
    c0 = 0.1
    aug = ExaPF.AugLagEvaluator(nlp, u0; c₀=c0, scale=true)
    solution = ExaPF.optimize!(algo, aug, u0)
    return solution
end

datafile = joinpath(dirname(@__FILE__), "..", "..", "ExaPF.jl", "data", "case57.m")
# datafile = "../pglib-opf/pglib_opf_case57_ieee.m"
# datafile = "../../pglib-opf/pglib_opf_case118_ieee.m"
# datafile = "../pglib-opf/pglib_opf_case1354_pegase.m"
# datafile = "../pglib-opf/pglib_opf_case9241_pegase.m"

nlp = ExaPF.ReducedSpaceEvaluator(datafile)
u0 = ExaPF.initial(nlp)
solution = @time auglag(nlp, u0; itout_max=20, iter_max=1000)

solution.minimizer

