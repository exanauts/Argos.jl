
using MadNLP


function solve_auglag_madnlp(aug; linear_solver=MadNLPLapackCPU, max_iter=20, penalty=0.1, rate=10.0)
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
    mnlp = MadNLP.NonlinearProgram(aug)
    madnlp_options = Dict{Symbol, Any}(:tol=>1e-5,
                                :kkt_system=>MadNLP.DENSE_KKT_SYSTEM,
                                :linear_solver=>linear_solver,
                                :print_level=>MadNLP.ERROR)
    ipp = MadNLP.Solver(mnlp; option_dict=madnlp_options)
    solver = ExaOpt.AuglagSolver(ipp, options)

    x0 = ExaOpt.initial(aug)
    aug.ρ = penalty # update penalty in Evaluator

    ExaOpt.optimize!(solver, aug, x0)
    return ipp
end

function test_dense(aug)
    ExaOpt.reset!(aug)
    mnlp = MadNLP.NonlinearProgram(aug)
    options = Dict{Symbol, Any}(:tol=>1e-5, :max_iter=>100,
                                :kkt_system=>MadNLP.DENSE_KKT_SYSTEM,
                                :linear_solver=>MadNLPLapackCPU)
    ipp = MadNLP.Solver(mnlp; option_dict=options)
    MadNLP.optimize!(ipp)
    return ipp
end

function test_sparse(aug)
    ExaOpt.reset!(aug)
    mnlp = MadNLP.NonlinearProgram(aug; allocate_buffer=true)
    options = Dict{Symbol, Any}(:tol=>1e-5, :max_iter=>30,
                                :linear_solver=>MadNLPLapackCPU)
    ips = MadNLP.Solver(mnlp; option_dict=options)
    MadNLP.optimize!(ips)
    return ips
end

# TODO: update
function test_dense_gpu(aug)
    ExaOpt.reset!(aug)
    mnlp = MadNLP.NonlinearProgram(aug)
    options = Dict{Symbol, Any}(:tol=>1e-5, :max_iter=>30,
                                :linear_solver=>MadNLPLapackGPU)
    ipp = MadNLP.DenseSolver(mnlp, CuVector{Float64}, CuMatrix{Float64};
                             option_dict=options)
    # warmstart
    ipp.lag_hess!(ipp.x, ipp.l)
    ipp.cnt.start_time = time()
    MadNLP.optimize!(ipp)
    return ipp
end
