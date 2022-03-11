
using MadNLP

function _test_results_match(ips1, ips2; atol=1e-10)
    @test ips1.status == ips2.status
    @test ips1.cnt.k == ips2.cnt.k
    @test ips1.obj_val ≈ ips2.obj_val atol=atol
    @test ips1.x ≈ ips2.x atol=atol
    @test ips1.l ≈ ips2.l atol=atol
end

# Solve with default options (reference).
function _madnlp_default(nlp; kwargs...)
    Argos.reset!(nlp)
    options = Dict{Symbol, Any}(kwargs...)
    mnlp = Argos.OPFModel(nlp)
    ips = MadNLP.InteriorPointSolver(mnlp; option_dict=options)
    MadNLP.optimize!(ips)
    return ips
end

# Solve with DENSE_KKT_SYSTEM
function _madnlp_dense_kkt(nlp; kwargs...)
    Argos.reset!(nlp)
    options = Dict{Symbol, Any}(kwargs...)
    options[:kkt_system] = MadNLP.DENSE_KKT_SYSTEM
    options[:linear_solver] = MadNLP.MadNLPLapackCPU
    options[:dual_initialized] = true
    mnlp = Argos.OPFModel(nlp)
    ipd = MadNLP.InteriorPointSolver(mnlp; option_dict=options)
    MadNLP.optimize!(ipd)
    return ipd
end

# Solve with DENSE_CONDENSED_KKT_SYSTEM
function _madnlp_condensed_kkt(nlp; kwargs...)
    Argos.reset!(nlp)
    options = Dict{Symbol, Any}(kwargs...)
    options[:kkt_system] = MadNLP.DENSE_CONDENSED_KKT_SYSTEM
    options[:linear_solver] = MadNLP.MadNLPLapackCPU
    options[:dual_initialized] = true
    mnlp = Argos.OPFModel(nlp)
    ipc = MadNLP.InteriorPointSolver(mnlp; option_dict=options)
    MadNLP.optimize!(ipc)
    return ipc
end

# Solve with BieglerKKTSystem
function _madnlp_biegler_kkt(nlp; kwargs...)
    Argos.reset!(nlp)
    options_biegler = Dict{Symbol, Any}(kwargs...)
    options_biegler[:linear_solver] = MadNLP.MadNLPLapackCPU
    madopt = MadNLP.Options(linear_solver=MadNLP.MadNLPLapackCPU)
    MadNLP.set_options!(madopt, options_biegler, Dict())
    KKT = Argos.BieglerKKTSystem{Float64, Vector{Int}, Vector{Float64}, Matrix{Float64}}
    mnlp = Argos.OPFModel(nlp)
    ipb = MadNLP.InteriorPointSolver{KKT}(mnlp, madopt; option_linear_solver=options_biegler)
    MadNLP.optimize!(ipb)
    return ipb
end

@testset "MadNLP wrapper" begin
    datafile = joinpath(INSTANCES_DIR, "case30.m")
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

