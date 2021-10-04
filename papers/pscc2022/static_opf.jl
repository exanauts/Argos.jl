using ExaPF
using MadNLP
using ExaOpt
using CUDA

if CUDA.has_cuda_gpu()
    using CUDAKernels
    using CUDA.CUSPARSE
    using MadNLPGPU
    # Include CUDA extension for ExaOpt
    include(joinpath(dirname(pathof(ExaOpt)), "..", "test", "cusolver.jl"))
end

const DATA_DIRECTORY = joinpath(dirname(pathof(ExaOpt)), "..", "data")


function solve_auglag_madnlp(aug; linear_solver=MadNLPLapackCPU, max_iter=10, penalty=10.0, rate=10.0)
    # Clean AugLagEvaluator
    ExaOpt.reset!(aug)
    aug.ρ = penalty # update penalty in Evaluator
    # MadNLP solver
    mnlp = MadNLP.NonlinearProgram(aug)
    madnlp_options = Dict{Symbol, Any}(
        :tol=>1e-5,
        :kkt_system=>MadNLP.DENSE_KKT_SYSTEM,
        :linear_solver=>linear_solver,
        :print_level=>MadNLP.ERROR,
    )
    ipp = MadNLP.Solver(mnlp; option_dict=madnlp_options)
    # Auglag solver
    solver = ExaOpt.AuglagSolver(ipp; max_iter=max_iter, verbose=1, rate=rate, ε_dual=1e-2, ε_primal=1e-5)

    x0 = ExaOpt.initial(aug)

    return ExaOpt.optimize!(solver, aug, x0)
end

function solve_auglag_madnlp_schur(aug; linear_solver=MadNLPLapackGPU, max_iter=10, penalty=10.0, rate=10.0)
    @assert CUDA.has_cuda_gpu()
    # Clean AugLagEvaluator
    ExaOpt.reset!(aug)
    aug.ρ = penalty # update penalty in Evaluator
    # MadNLP solver
    mnlp = MadNLP.NonlinearProgram(aug)
    madnlp_options = Dict{Symbol, Any}(
        :tol=>1e-5,
        :kkt_system=>MadNLP.DENSE_KKT_SYSTEM,
        :linear_solver=>linear_solver,
        :print_level=>MadNLP.ERROR)
    kkt = ExaOpt.MixedAuglagKKTSystem{Float64, CuVector{Float64}, CuMatrix{Float64}}(aug, Int[])
    ipp = MadNLP.Solver(mnlp; option_dict=madnlp_options, kkt=kkt)
    # Auglag solver
    solver = ExaOpt.AuglagSolver(ipp; max_iter=max_iter, verbose=1, rate=rate, ε_dual=1e-2, ε_primal=1e-5)

    x0 = ExaOpt.initial(aug)

    return ExaOpt.optimize!(solver, aug, x0)
end

function _set_manual_scaler!(aug)
    nlp = aug.inner
    g_min, g_max = ExaOpt.bounds(nlp, ExaOpt.Constraints())
    aug.scaler = ExaOpt.NetworkScaler(nlp, g_min, g_max)
    return
end

"""
    Used to reproduce results displayed in Table II, page 6 (need a GPU)
"""
function pscc_solve_static_opf()
    @assert CUDA.has_cuda_gpu()
    nbatches = 250
    for case in [
        "case1354pegase.m",
        "case2869pegase.m",
        "case9241pegase.m",
    ]
        datafile = joinpath(DATA_DIRECTORY, case)
        # Instantiate auglag problem
        aug_g = ExaOpt.instantiate_auglag_model(
            datafile;
            scale=false, line_constraints=true,
            device=CUDADevice(), nbatches=nbatches,
        )
        # Scale the problem manually
        _set_manual_scaler!(aug_g)
        # Solve!
        res = solve_auglag_madnlp_schur(aug_g; rate=5.0, max_iter=20)
    end
end

