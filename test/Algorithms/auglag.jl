@testset "Augmented Lagrangian solver" begin
    datafile = joinpath(INSTANCES_DIR, "case9.m")
    nlp = Argos.ReducedSpaceEvaluator(datafile; auglag=true)
    aug = Argos.AugLagEvaluator(nlp)
    model = Argos.OPFModel(aug)
    subsolver = MadNLP.MadNLPSolver(
        model;
        tol=1e-5,
        kkt_system=MadNLP.DENSE_KKT_SYSTEM,
        linear_solver=LapackCPUSolver,
        print_level=MadNLP.ERROR,
    )
    solver = Argos.AuglagSolver(subsolver; max_iter=3, verbose=0)
    stats = Argos.optimize!(solver, aug, x0)
    @test stats.minimum ≈ 5295.8 rtol=1e-3
end

