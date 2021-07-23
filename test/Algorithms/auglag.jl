@testset "Augmented Lagrangian solver" begin
    datafile = joinpath(dirname(pathof(ExaPF)), "..", "data", "case57.m")
    nlp = ExaOpt.ReducedSpaceEvaluator(datafile)

    @testset "NGPA backend" begin
        algo = ExaOpt.AugLagSolver(;
            max_iter=20,
            verbose=0,
            ωtol=1e-5,
            scaling=true,
            inner_algo=:ngpa,
            ε_dual=1e-2,
        )
        solution = ExaOpt.optimize!(algo, nlp, ExaOpt.initial(nlp))
        @test solution.status == MOI.ITERATION_LIMIT
        # NGPA slow to converges (no second order information)
        # Solution is larger than expected.
        @test solution.minimum ≈ 3.7593e+04 rtol=1e-3
    end
end
