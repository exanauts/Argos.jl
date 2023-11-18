using DelimitedFiles
using LazyArtifacts
using LinearAlgebra
using Printf
using Random

using NLPModels
using Argos
using ExaPF
using MadNLP

# HSL
using MadNLPHSL

# GPU
using CUDA
using KernelAbstractions
using ArgosCUDA
using MadNLPGPU

const PS = ExaPF.PowerSystem

const DATA = joinpath(artifact"ExaData", "ExaData")
RESULTS_DIR = "results"

if CUDA.has_cuda()
    CUDA.allowscalar(false)
end

function refresh_memory()
    GC.gc(true)
    CUDA.has_cuda() && CUDA.reclaim()
    return
end

function init_model!(blk)
    x0 = NLPModels.get_x0(blk)
    nnzj = NLPModels.get_nnzj(blk)
    jac = zeros(nnzj)
    NLPModels.jac_coord!(blk, x0, jac)
    return
end

function build_opf_model(model; use_gpu=false)
    if use_gpu
        model_gpu = PolarForm(model, CUDABackend())
        nlp = Argos.FullSpaceEvaluator(model_gpu)
        blk = Argos.OPFModel(Argos.bridge(nlp))
    else
        nlp = Argos.FullSpaceEvaluator(model)
        blk = Argos.OPFModel(nlp)
    end
    init_model!(blk)
    return blk
end

function build_madnlp(
    blk::Argos.OPFModel,
    ::Argos.FullSpace;
    max_iter=max_iter,
    dual_initialized=true,
    tol=1e-5,
    print_level=MadNLP.ERROR,
    linear_solver=Ma27Solver,
)
    return MadNLP.MadNLPSolver(blk; max_iter=max_iter, dual_initialized=dual_initialized, tol=tol, print_level=print_level, linear_solver=linear_solver)
end

function build_madnlp(
    blk::Argos.OPFModel,
    ::Argos.BieglerReduction;
    max_iter=max_iter,
    dual_initialized=true,
    tol=1e-5,
    print_level=MadNLP.ERROR,
    linear_solver=nothing,
)
    madnlp_options = Dict{Symbol, Any}()
    madnlp_options[:linear_solver] = LapackGPUSolver
    madnlp_options[:lapack_algorithm] = MadNLP.CHOLESKY
    madnlp_options[:dual_initialized] = dual_initialized
    madnlp_options[:max_iter] = max_iter
    madnlp_options[:print_level] = print_level
    madnlp_options[:tol] = tol
    opt_ipm, opt_linear, logger = MadNLP.load_options(; madnlp_options...)
    KKT = Argos.BieglerKKTSystem{Float64, CuVector{Int}, CuVector{Float64}, CuMatrix{Float64}}
    return MadNLP.MadNLPSolver{Float64, KKT}(blk, opt_ipm, opt_linear; logger=logger)
end

