
using MadNLP


function solve_auglag_madnlp(aug; linear_solver=MadNLPLapackGPU, max_iter=20, penalty=0.1, rate=10.0)
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
    # aug.tracker = ExaOpt.NLPTracker(aug)
    mnlp = MadNLP.NonlinearProgram(aug)
    madnlp_options = Dict{Symbol, Any}(:tol=>1e-5,
                                       :kkt_system=>MadNLP.DENSE_KKT_SYSTEM,
                                       :linear_solver=>linear_solver,
                                       :print_level=>MadNLP.INFO)
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
    CUDA.@time MadNLP.optimize!(ipp)
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
    CUDA.@time MadNLP.optimize!(ipp)
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
    CUDA.@time MadNLP.optimize!(ipp)
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
    CUDA.@time MadNLP.optimize!(ipp)
    return ipp
end
