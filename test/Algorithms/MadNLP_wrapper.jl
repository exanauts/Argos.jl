
using MadNLP
using MadNLPTests

if CUDA.has_cuda_gpu()
    using MadNLPGPU
end

function _test_results_match(solver1, solver2; atol=1e-10)
    @test solver1.status == solver2.status
    @test solver1.cnt.k == solver2.cnt.k
    @test solver1.obj_val ≈ solver2.obj_val atol=atol
    @test MadNLP.primal(solver1.x) ≈ MadNLP.primal(solver2.x) atol=atol
    @test solver1.y ≈ solver2.y atol=atol
end

# Solve with default options (reference).
function _madnlp_default(nlp; kwargs...)
    Argos.reset!(nlp)
    mnlp = Argos.OPFModel(nlp)
    solver = MadNLP.MadNLPSolver(mnlp; kwargs...)
    MadNLP.solve!(solver)
    return solver
end

# Solve with DENSE_KKT_SYSTEM
function _madnlp_dense_kkt(nlp; kwargs...)
    Argos.reset!(nlp)
    mnlp = Argos.OPFModel(nlp)
    solver = MadNLP.MadNLPSolver(
        mnlp;
        kkt_system=MadNLP.DenseKKTSystem,
        linear_solver=LapackCPUSolver,
        kwargs...
    )
    MadNLP.solve!(solver)
    return solver
end

# Solve with DENSE_CONDENSED_KKT_SYSTEM
function _madnlp_condensed_kkt(nlp; kwargs...)
    Argos.reset!(nlp)
    mnlp = Argos.OPFModel(nlp)
    solver = MadNLP.MadNLPSolver(
        mnlp;
        kkt_system=MadNLP.DenseCondensedKKTSystem,
        linear_solver=LapackCPUSolver,
        kwargs...
    )
    MadNLP.solve!(solver)
    return solver
end

# Solve with BieglerKKTSystem
function _madnlp_biegler_kkt(nlp; kwargs...)
    Argos.reset!(nlp)
    mnlp = Argos.OPFModel(nlp)

    T = Float64
    VI = Vector{Int}
    VT = Vector{T}
    MT = Matrix{T}
    solver = MadNLP.MadNLPSolver(
        mnlp;
        kkt_system=Argos.BieglerKKTSystem{T, VI, VT, MT},
        linear_solver=LapackCPUSolver,
        callback=MadNLP.SparseCallback,
        kwargs...
    )
    MadNLP.solve!(solver)
    return solver
end

@testset "[CPU] BieglerKKTSystem" begin
    case = "case9.m"
    datafile = joinpath(INSTANCES_DIR, case)
    opf = Argos.FullSpaceEvaluator(datafile)

    T = Float64
    VI = Array{Int, 1}
    VT = Array{T, 1}
    MT = Array{T, 2}

    linear_solver = LapackCPUSolver
    options = MadNLP.MadNLPOptions(; linear_solver=linear_solver)
    options_linear_solver = MadNLP.LapackOptions(
        lapack_algorithm=MadNLP.LU,
    )
    cnt = MadNLP.MadNLPCounters(; start_time=time())

    nlp = Argos.OPFModel(Argos.bridge(opf))
    ind_cons = MadNLP.get_index_constraints(
        nlp;
        equality_treatment=MadNLP.EnforceEquality,
    )

    cb = MadNLP.create_callback(
        MadNLP.SparseCallback,
        nlp;
    )
    kkt = MadNLP.create_kkt_system(
        Argos.BieglerKKTSystem{T, VI, VT, MT},
        cb,
        ind_cons,
        linear_solver;
        opt_linear_solver=options_linear_solver,
    )
    MadNLPTests.test_kkt_system(kkt, cb)
end

@testset "[CPU] MadNLP wrapper: $case" for case in [
    "case30.m",
    "case57.m",
]
    tol = 1e-6
    datafile = joinpath(INSTANCES_DIR, case)
    options = Dict{Symbol, Any}(
        :dual_initialized=>true,
        :tol=>tol,
        :print_level=>MadNLP.ERROR,
    )
    @testset "Reduce-then-linearize" begin
        nlp = Argos.ReducedSpaceEvaluator(datafile)
        solver = _madnlp_default(nlp; options...)
        @test solver.status == MadNLP.SOLVE_SUCCEEDED
        ipd = _madnlp_dense_kkt(nlp; options...)
        _test_results_match(solver, ipd; atol=tol)
        ipc = _madnlp_condensed_kkt(nlp; options...)
        _test_results_match(solver, ipc; atol=tol)
    end
    @testset "Linearize-then-reduce" begin
        flp = Argos.FullSpaceEvaluator(datafile)
        solver = _madnlp_default(flp; options...)
        @test solver.status == MadNLP.SOLVE_SUCCEEDED
        ipb = _madnlp_biegler_kkt(flp; options...)
        _test_results_match(solver, ipb; atol=tol)
        @test ipb.kkt.Wref === flp.hess.H
    end
end

@testset "[CPU] Solve OPF with $form" for (form, linear_solver_algo) in [
    (Argos.FullSpace(), MadNLP.BUNCHKAUFMAN),
    (Argos.BieglerReduction(), MadNLP.CHOLESKY),
    (Argos.DommelTinney(), MadNLP.CHOLESKY),
]
    case = "case9.m"
    datafile = joinpath(INSTANCES_DIR, case)

    solver = Argos.run_opf(
        datafile,
        form;
        tol=1e-5,
        print_level=MadNLP.ERROR,
        linear_solver=LapackCPUSolver,
        lapack_algorithm=linear_solver_algo,
    )
    @test isa(solver, MadNLP.MadNLPSolver)
    @test solver.status == MadNLP.SOLVE_SUCCEEDED
end

if CUDA.has_cuda_gpu()
    # Test BieglerKKTSystem on the GPU.
    @testset "[CUDA] BieglerKKTSystem" begin
        case = "case9.m"
        datafile = joinpath(INSTANCES_DIR, case)
        opf = Argos.FullSpaceEvaluator(datafile)
        solver_ref = _madnlp_default(opf; print_level=MadNLP.ERROR)

        opf_gpu = Argos.FullSpaceEvaluator(datafile; device=CUDABackend())
        KKT = Argos.BieglerKKTSystem{Float64, CuVector{Int}, CuVector{Float64}, CuMatrix{Float64}}
        nlp = Argos.OPFModel(Argos.bridge(opf_gpu))
        solver_gpu = MadNLP.MadNLPSolver(
            nlp;
            kkt_system=KKT,
            linear_solver=LapackGPUSolver,
            lapack_algorithm=MadNLP.CHOLESKY,
            callback=MadNLP.SparseCallback,
            equality_treatment=MadNLP.EnforceEquality,
            print_level=MadNLP.ERROR,
        )
        stats = MadNLP.solve!(solver_gpu)
        _test_results_match(solver_ref, solver_gpu; atol=1e-6)
    end

    @testset "[CUDA] Solve OPF with $form" for (form, linear_solver, linear_solver_algo) in [
        (Argos.FullSpace(), LapackCPUSolver, MadNLP.BUNCHKAUFMAN),
        (Argos.BieglerReduction(), LapackGPUSolver, MadNLP.CHOLESKY),
        (Argos.DommelTinney(), LapackGPUSolver, MadNLP.CHOLESKY),
    ]
        case = "case9.m"
        datafile = joinpath(INSTANCES_DIR, case)

        solver = Argos.run_opf_gpu(
            datafile,
            form;
            tol=1e-5,
            linear_solver=linear_solver,
            lapack_algorithm=linear_solver_algo,
            print_level=MadNLP.ERROR,
            max_iter=100,
        )
        @test isa(solver, MadNLP.MadNLPSolver)
        @test solver.status == MadNLP.SOLVE_SUCCEEDED
    end
end

