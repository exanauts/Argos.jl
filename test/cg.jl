
datafile = joinpath(dirname(@__FILE__), "..", "..", "ExaPF.jl", "data", "case57.m")
# datafile = joinpath(dirname(@__FILE__), "..", "..", "ExaPF.jl", "data", "case300.m")
# datafile = joinpath(dirname(@__FILE__), "..", "..", "ExaPF.jl", "data", "case9.m")
# datafile = "../../pglib-opf/pglib_opf_case118_ieee.m"
# datafile = "../../pglib-opf/pglib_opf_case1354_pegase.m"
nlp = ExaPF.ReducedSpaceEvaluator(datafile)
u0 = ExaPF.initial(nlp)
aug = ExaPF.AugLagEvaluator(nlp, u0; c₀=1e-1, scale=true)

algo = Argos.HZ()
# res1 = Argos.optimize(algo, aug, u0)

α0 = 1e-0
res2 = @time Argos.ngpa(aug, u0; tol=1e-5, max_iter=2_000, α♯=α0, α_bb=α0)
a1 = aug.counter.objective

u0 = ExaPF.initial(nlp)
res3 = @time Argos.asa(aug, u0; α♯=α0, max_iter=100, gp_tol=1e-4, cg_tol=1e-5)
a2 = aug.counter.objective

