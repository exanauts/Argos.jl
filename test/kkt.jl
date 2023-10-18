
using Test
using LazyArtifacts
using LinearAlgebra

using MadNLP
using NLPModels
using Argos

const INSTANCES_DIR = joinpath(artifact"ExaData", "ExaData")

function test_biegler_kkt(casename)
    datafile = joinpath(INSTANCES_DIR, casename)
    nlp = Argos.FullSpaceEvaluator(datafile)
    opf = Argos.OPFModel(nlp)

    # Init OPF structure
    x0 = NLPModels.get_x0(opf)
    nnzj = NLPModels.get_nnzj(opf)
    jac = zeros(nnzj)
    NLPModels.jac_coord!(opf, x0, jac)

    # Problem's dimensions
    nx = nlp.nx
    nu = nlp.nu
    n, m = NLPModels.get_nvar(opf), NLPModels.get_ncon(opf)
    ind_cons = MadNLP.get_index_constraints(opf)
    n_ineq = length(ind_cons.ind_ineq)

    # Build reference KKT system
    T = Float64
    VI = Vector{Int}
    VT = Vector{Float64}
    MT = Matrix{Float64}

    # Build KKT system
    KKT = Argos.BieglerKKTSystem{T, VI, VT, MT}
    kkt = KKT(opf, ind_cons)
    MadNLP.initialize!(kkt)
    MadNLP.build_kkt!(kkt)

    # Test sizes are matching.
    @test size(kkt.Wref) == (n, n)
    @test length(kkt.pr_diag) == n + n_ineq
    @test size(kkt.Gx) == (nx, nx)
    @test size(kkt.Gu) == (nx, nu)
    @test size(kkt.A)  == (n_ineq, nx + nu)
    @test size(kkt.aug_com) == (nu, nu)

    @test kkt.Wref === nlp.hess.H

    # Test with MadNLP
    madnlp_options = Dict{Symbol, Any}(
        :lapack_algorithm=>MadNLP.CHOLESKY,
        :linear_solver=>LapackCPUSolver,
    )
    opt_ipm, opt_linear, logger = MadNLP.load_options(; madnlp_options...)
    solver = MadNLP.MadNLPSolver{T, KKT}(opf, opt_ipm, opt_linear; logger=logger)

    @test isa(solver.kkt, KKT)
    # Test building KKT within MadNLP.
    MadNLP.initialize!(solver.kkt)
    MadNLP.build_kkt!(solver.kkt)

    # Load non-trivial values in KKT system.
    MadNLP.eval_jac_wrapper!(solver, solver.kkt, solver.x)
    MadNLP.eval_grad_f_wrapper!(solver, solver.f, solver.x)
    MadNLP.set_initial_rhs!(solver, solver.kkt)

    # Solve KKT system.
    MadNLP.factorize_wrapper!(solver)
    MadNLP.solve_refine_wrapper!(solver, solver.d, solver.p)

    # Test solution is correct
    p_ref = solver.p.values[1:n+n_ineq+m]
    d = solver.d.values[1:n+n_ineq+m]
    p = zeros(length(d))
    mul!(p, solver.kkt, d)
    @test isapprox(p, p_ref)
    return
end

@testset "BieglerKKTSystem $case" for case in ["case9.m", "case30.m"]
    test_biegler_kkt(case)
end

