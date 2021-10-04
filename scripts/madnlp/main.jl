
using Revise
using ExaPF
using MadNLP
using MathOptInterface
using ExaOpt
using CUDA

const MOI = MathOptInterface

# MadNLP wrapper
include(joinpath(dirname(@__FILE__), "..", "common.jl"))
include(joinpath(dirname(@__FILE__), "..", "problems.jl"))

function madnlp_subproblem(aug; linear_solver=MadNLPLapackCPU, max_iter=100)
    ExaOpt.reset!(aug)
    optimizer = MadNLP.Optimizer(linear_solver=linear_solver)
    MOI.set(optimizer, MOI.RawParameter("tol"), 1e-5)
    MOI.set(optimizer, MOI.RawParameter("print_level"), MadNLP.DEBUG)
    MOI.set(optimizer, MOI.RawParameter("max_iter"), max_iter)
    solution = @time ExaOpt.optimize!(optimizer, aug)

    return optimizer.ips
end

function solve_auglag_moi(aug; linear_solver=MadNLPLapackCPU, max_iter=20, penalty=0.1, rate=10.0)
    ExaOpt.reset!(aug)
    aug.tracker = ExaOpt.NLPTracker(aug)
    options = ExaOpt.AugLagOptions(;
        max_iter=max_iter,
        max_inner_iter=1000,
        α0=1.0,
        rate=rate,
        ωtol=1e-6,
        verbose=1,
        ε_dual=1e-2,
        ε_primal=1e-5,
    )
    optimizer = MadNLP.Optimizer(
        linear_solver=linear_solver, print_level=MadNLP.ERROR, max_iter=500,
    )
    solver = ExaOpt.AuglagSolver(optimizer, options)

    x0 = ExaOpt.initial(aug)
    aug.ρ = penalty # update penalty in Evaluator

    return ExaOpt.optimize!(solver, aug, x0)
end

function solve_auglag_madnlp(aug; linear_solver=MadNLPLapackCPU, max_iter=10, penalty=10.0, rate=10.0)
    options = ExaOpt.AugLagOptions(;
        max_iter=max_iter,
        max_inner_iter=100,
        α0=1.0,
        rate=rate,
        ωtol=1e-5,
        verbose=1,
        ε_dual=1e-2,
        ε_primal=1e-5,
    )
    ExaOpt.reset!(aug)
    aug.ρ = penalty # update penalty in Evaluator
    mnlp = MadNLP.NonlinearProgram(aug)
    madnlp_options = Dict{Symbol, Any}(
        :tol=>1e-5,
        :kkt_system=>MadNLP.DENSE_KKT_SYSTEM,
        :linear_solver=>linear_solver,
        :print_level=>MadNLP.ERROR,
    )
    ipp = MadNLP.Solver(mnlp; option_dict=madnlp_options)
    solver = ExaOpt.AuglagSolver(ipp, options)

    x0 = ExaOpt.initial(aug)

    return ExaOpt.optimize!(solver, aug, x0)
end

function solve_auglag_madnlp_schur(aug; linear_solver=MadNLPLapackCPU, max_iter=10, penalty=10.0, rate=10.0)
    options = ExaOpt.AugLagOptions(;
        max_iter=max_iter,
        max_inner_iter=100,
        α0=1.0,
        rate=rate,
        ωtol=1e-5,
        verbose=1,
        ε_dual=1e-2,
        ε_primal=1e-5,
    )
    ExaOpt.reset!(aug)
    mnlp = MadNLP.NonlinearProgram(aug)
    madnlp_options = Dict{Symbol, Any}(:tol=>1e-5,
                                       :kkt_system=>MadNLP.DENSE_KKT_SYSTEM,
                                       :linear_solver=>linear_solver,
                                       :print_level=>MadNLP.ERROR)
    kkt = ExaOpt.MixedAuglagKKTSystem{Float64, CuVector{Float64}, CuMatrix{Float64}}(aug, Int[])
    ipp = MadNLP.Solver(mnlp; option_dict=madnlp_options, kkt=kkt)
    solver = ExaOpt.AuglagSolver(ipp, options)

    x0 = ExaOpt.initial(aug)
    aug.ρ = penalty # update penalty in Evaluator

    return ExaOpt.optimize!(solver, aug, x0)
end

function test_dense(
    aug; max_iter=100, scaling=true,
    linear_solver=MadNLPLapackCPU,
    inertia=MadNLP.INERTIA_AUTO,
)
    ExaOpt.reset!(aug)
    mnlp = MadNLP.NonlinearProgram(aug)
    options = Dict{Symbol, Any}(
        :tol=>1e-5, :max_iter=>max_iter,
        :nlp_scaling=>scaling,
        :inertia_correction_method=>inertia,
        :kkt_system=>MadNLP.DENSE_KKT_SYSTEM,
        :print_level=>MadNLP.DEBUG,
        :linear_solver=>linear_solver
    )
    ipp = MadNLP.Solver(mnlp; option_dict=options)
    @time MadNLP.optimize!(ipp)
    return ipp
end

function test_dense_new(
    aug; max_iter=100, scaling=true,
    linear_solver=MadNLPLapackCPU,
    inertia=MadNLP.INERTIA_AUTO,
)
    ExaOpt.reset!(aug)
    mnlp = MadNLP.NonlinearProgram(aug)
    options = Dict{Symbol, Any}(
        :tol=>1e-5, :max_iter=>max_iter,
        :nlp_scaling=>scaling,
        :inertia_correction_method=>inertia,
        :print_level=>MadNLP.DEBUG,
        :kkt_system=>MadNLP.DENSE_KKT_SYSTEM,
        :linear_solver=>linear_solver
    )
    # Custom KKT system
    kkt = ExaOpt.MixedAuglagKKTSystem{Float64, Vector{Float64}, Matrix{Float64}}(aug, Int[])
    ipp = MadNLP.Solver(mnlp; option_dict=options, kkt=kkt)
    @time MadNLP.optimize!(ipp)
    return ipp
end

function test_dense_gpu(aug; max_iter=100)
    ExaOpt.reset!(aug)
    mnlp = MadNLP.NonlinearProgram(aug)
    n = ExaOpt.n_variables(aug)
    # Instantiate KKT system on the device
    kkt = MadNLP.DenseKKTSystem{Float64, CuVector{Float64}, CuMatrix{Float64}}(mnlp; buffer_size=n)
    options = Dict{Symbol, Any}(:tol=>1e-5, :max_iter=>max_iter,
                                :print_level=>MadNLP.DEBUG,
                                :kkt_system=>MadNLP.DENSE_KKT_SYSTEM,
                                :linear_solver=>MadNLPLapackGPU)
    ipp = MadNLP.Solver(mnlp; kkt=kkt, option_dict=options)
    # warmstart
    MadNLP.eval_lag_hess_wrapper!(ipp, kkt, ipp.x, ipp.l)

    ipp.cnt = MadNLP.Counters(start_time=time())
    @time MadNLP.optimize!(ipp)
    return ipp
end

function test_dense_gpu_new(aug; max_iter=100)
    ExaOpt.reset!(aug)
    mnlp = MadNLP.NonlinearProgram(aug)
    n = ExaOpt.n_variables(aug)
    # Instantiate KKT system on the device
    kkt = ExaOpt.MixedAuglagKKTSystem{Float64, CuVector{Float64}, CuMatrix{Float64}}(aug, Int[])
    options = Dict{Symbol, Any}(:tol=>1e-5, :max_iter=>max_iter,
                                :print_level=>MadNLP.DEBUG,
                                :kkt_system=>MadNLP.DENSE_KKT_SYSTEM,
                                :linear_solver=>MadNLPLapackGPU)
    ipp = MadNLP.Solver(mnlp; kkt=kkt, option_dict=options)
    # warmstart
    MadNLP.eval_lag_hess_wrapper!(ipp, kkt, ipp.x, ipp.l)

    ipp.cnt = MadNLP.Counters(start_time=time())
    MadNLP.optimize!(ipp)
    return ipp
end

