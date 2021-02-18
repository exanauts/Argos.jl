
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

    @testset "Tron backend" begin
        algo = ExaOpt.AugLagSolver(;
            max_iter=20,
            verbose=0,
            ωtol=1e-5,
            scaling=true,
            inner_algo=:tron,
            ε_dual=1e-2,
        )
        # Test first Tron directly on nlp, with discontinuous
        # second order derivatives. Algorithm is unable to converge.
        solution = ExaPF.optimize!(algo, nlp, ExaPF.initial(nlp))
        @test solution.status == MOI.ITERATION_LIMIT
        @test solution.minimum ≈ 3.7589338e+04 rtol=1e-6
        # Reformulate inequalities with slacks to
        # get continuous second order derivatives.
        slk = ExaPF.SlackEvaluator(nlp)
        solution = ExaPF.optimize!(algo, slk, ExaPF.initial(slk))
        @test solution.status == MOI.OPTIMAL
        @test solution.minimum ≈ 3.7589338e+04 rtol=1e-8
    end
end
