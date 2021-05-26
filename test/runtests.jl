
using Test
using ExaOpt
using ExaPF
using MathOptInterface

const MOI = MathOptInterface

const CASE = joinpath(dirname(@__FILE__), "case57.m")

@testset "Augmented Lagrangian solver" begin
    nlp = ExaPF.ReducedSpaceEvaluator(CASE)

    @testset "NGPA backend" begin
        algo = ExaOpt.AugLagSolver(;
            max_iter=20,
            verbose=0,
            ωtol=1e-5,
            scaling=true,
            inner_algo=:ngpa,
            ε_dual=1e-2,
        )
        solution = ExaPF.optimize!(algo, nlp, ExaPF.initial(nlp))
        @test solution.status == MOI.ITERATION_LIMIT
        # NGPA slow to converges (no second order information)
        # Solution is larger than expected.
        @test solution.minimum ≈ 3.7593e+04 rtol=1e-3
    end
end
