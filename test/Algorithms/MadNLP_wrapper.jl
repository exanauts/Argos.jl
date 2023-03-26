
using MadNLP

function _test_results_match(ips1, ips2; atol=1e-10)
    @test ips1.status == ips2.status
    @test ips1.cnt.k == ips2.cnt.k
    @test ips1.obj_val ≈ ips2.obj_val atol=atol
    @test MadNLP.primal(ips1.x) ≈ MadNLP.primal(ips2.x) atol=atol
    @test ips1.y ≈ ips2.y atol=atol
end

# Solve with default options (reference).
function _madnlp_default(nlp; kwargs...)
    Argos.reset!(nlp)
    mnlp = Argos.OPFModel(nlp)
    ips = MadNLP.MadNLPSolver(mnlp; kwargs...)
    MadNLP.solve!(ips)
    return ips
end

# Solve with DENSE_KKT_SYSTEM
function _madnlp_dense_kkt(nlp; kwargs...)
    Argos.reset!(nlp)
    mnlp = Argos.OPFModel(nlp)
    ipd = MadNLP.MadNLPSolver(
        mnlp;
        kkt_system=MadNLP.DENSE_KKT_SYSTEM,
        linear_solver=LapackCPUSolver,
        kwargs...
    )
    MadNLP.solve!(ipd)
    return ipd
end

# Solve with DENSE_CONDENSED_KKT_SYSTEM
function _madnlp_condensed_kkt(nlp; kwargs...)
    Argos.reset!(nlp)
    mnlp = Argos.OPFModel(nlp)
    ipc = MadNLP.MadNLPSolver(
        mnlp;
        kkt_system=MadNLP.DENSE_CONDENSED_KKT_SYSTEM,
        linear_solver=LapackCPUSolver,
        kwargs...
    )
    MadNLP.solve!(ipc)
    return ipc
end

# Solve with BieglerKKTSystem
function _madnlp_biegler_kkt(nlp; kwargs...)
    T = Float64
    VI = Vector{Int}
    VT = Vector{T}
    MT = Matrix{T}

    Argos.reset!(nlp)
    options_biegler = Dict{Symbol, Any}(kwargs...)
    options_biegler[:linear_solver] = LapackCPUSolver
    opt_ipm, opt_linear, logger = MadNLP.load_options(; options_biegler...)

    KKT = Argos.BieglerKKTSystem{T, VI, VT, MT}

    mnlp = Argos.OPFModel(nlp)
    ipb = MadNLP.MadNLPSolver{T, KKT}(mnlp, opt_ipm, opt_linear; logger=logger)
    MadNLP.solve!(ipb)
    return ipb
end

@testset "MadNLP wrapper: $case" for case in [
    "case30.m",
    "case57.m",
]
    datafile = joinpath(INSTANCES_DIR, case)
    options = Dict{Symbol, Any}(
        :dual_initialized=>true,
        :tol=>1e-6,
        :print_level=>MadNLP.ERROR,
    )
    @testset "Reduce-then-linearize" begin
        nlp = Argos.ReducedSpaceEvaluator(datafile)
        ips = _madnlp_default(nlp; options...)
        @test ips.status == MadNLP.SOLVE_SUCCEEDED
        ipd = _madnlp_dense_kkt(nlp; options...)
        _test_results_match(ips, ipd; atol=1e-8)
        ipc = _madnlp_condensed_kkt(nlp; options...)
        _test_results_match(ips, ipc; atol=1e-8)
    end
    @testset "Linearize-then-reduce" begin
        flp = Argos.FullSpaceEvaluator(datafile)
        ips = _madnlp_default(flp; options...)
        @test ips.status == MadNLP.SOLVE_SUCCEEDED
        ipb = _madnlp_biegler_kkt(flp; options...)
        _test_results_match(ips, ipb; atol=1e-8)
        @test ipb.kkt.Wref === flp.hess.H
    end
end

@testset "Solve OPF with $form" for form in [
    Argos.FullSpace(),
    Argos.BieglerReduction(),
    Argos.DommelTinney(),
]
    case = "case9.m"
    datafile = joinpath(INSTANCES_DIR, case)

    ips = Argos.run_opf(datafile, form; tol=1e-5, print_level=MadNLP.ERROR)
    @test isa(ips, MadNLP.MadNLPSolver)
    @test ips.status == MadNLP.SOLVE_SUCCEEDED
end

