
using Ipopt
using MathOptInterface

const MOI = MathOptInterface

@testset "MOI wrapper" begin
    CASE57_SOLUTION = [
        1.009292142238587,
        1.0075546562785322,
        1.0032634469498336,
        1.0256670737344367,
        1.0438264453243065,
        1.004075412844108,
        0.9918490001351085,
        0.8779133160884935,
        0.45070969015991846,
        0.7286288333432149,
        4.598049504568289,
        0.976335191632888,
        3.6152219445289338
    ]
    CASE57_OBJECTIVE = 41737.78674184994

    datafile = joinpath(INSTANCES_DIR, "case57.m")
    nlp = Argos.ReducedSpaceEvaluator(datafile; auglag=true)

    @testset "ReducedSpaceEvaluator with L-BFGS" begin
        optimizer = Ipopt.Optimizer()
        MOI.set(optimizer, MOI.RawParameter("print_level"), 0)
        MOI.set(optimizer, MOI.RawParameter("limited_memory_max_history"), 50)
        MOI.set(optimizer, MOI.RawParameter("hessian_approximation"), "limited-memory")
        MOI.set(optimizer, MOI.RawParameter("tol"), 1e-4)

        solution = Argos.optimize!(optimizer, nlp)
        MOI.empty!(optimizer)
        @test solution.status ∈ [MOI.OPTIMAL, MOI.LOCALLY_SOLVED]
        @test solution.minimum ≈ CASE57_OBJECTIVE
        @test solution.minimizer ≈ CASE57_SOLUTION rtol=1e-5
    end

    @testset "ReducedSpaceEvaluator with Hessian" begin
        optimizer = Ipopt.Optimizer()
        MOI.set(optimizer, MOI.RawParameter("print_level"), 0)
        MOI.set(optimizer, MOI.RawParameter("tol"), 1e-4)

        solution = Argos.optimize!(optimizer, nlp)
        MOI.empty!(optimizer)
        @test solution.status ∈ [MOI.OPTIMAL, MOI.LOCALLY_SOLVED]
        @test solution.minimum ≈ CASE57_OBJECTIVE
        @test solution.minimizer ≈ CASE57_SOLUTION rtol=1e-5
    end

    # Test resolution with AugLagEvaluator and Ipopt, as used inside ProxAL
    @testset "AugLagEvaluator with Hessian" begin
        optimizer = Ipopt.Optimizer()
        MOI.set(optimizer, MOI.RawParameter("print_level"), 0)
        MOI.set(optimizer, MOI.RawParameter("tol"), 1e-4)
        MOI.set(optimizer, MOI.RawParameter("max_iter"), 30)
        aug = Argos.AugLagEvaluator(nlp, Argos.initial(nlp); c₀=0.1)

        solution = Argos.optimize!(optimizer, aug)
        MOI.empty!(optimizer)
        @test solution.status ∈ [MOI.OPTIMAL, MOI.LOCALLY_SOLVED]
    end

    @testset "FullSpaceEvaluator with Hessian" begin
        optimizer = Ipopt.Optimizer()
        MOI.set(optimizer, MOI.RawParameter("print_level"), 0)
        MOI.set(optimizer, MOI.RawParameter("tol"), 1e-4)
        MOI.set(optimizer, MOI.RawParameter("max_iter"), 30)
        nlp = Argos.FullSpaceEvaluator(datafile; line_constraints=false)
        solution = Argos.optimize!(optimizer, nlp)
        MOI.empty!(optimizer)
        @test solution.status ∈ [MOI.OPTIMAL, MOI.LOCALLY_SOLVED]
        @test solution.minimum ≈ CASE57_OBJECTIVE
    end
end

