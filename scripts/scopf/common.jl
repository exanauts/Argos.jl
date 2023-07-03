using Dates
using DelimitedFiles
using Graphs
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
const MATPOWER = "/home/fpacaud/dev/matpower/data"
const SCENARIOS = joinpath(artifact"ExaData", "ExaData", "mp_demand")
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

function build_scopf_model(model, lines; use_gpu=false)
    contingencies = [ExaPF.LineContingency(l) for l in lines]
    nblks = length(contingencies) + 1
    if use_gpu
        model_gpu = PolarForm(model, CUDABackend())
        nlp = Argos.StochEvaluator(model_gpu, nblks; contingencies=contingencies)
        blk = Argos.OPFModel(Argos.bridge(nlp))
    else
        nlp = Argos.StochEvaluator(model, nblks; contingencies=contingencies)
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

# # Custom scaling
function MadNLP.scale_objective(nlp::Argos.OPFModel, grad::AbstractVector; max_gradient=1e-8)
    return min(1, max_gradient / norm(grad, Inf))
end

function MadNLP.scale_constraints!(
    nlp::Argos.OPFModel,
    con_scale::AbstractVector,
    jac::AbstractMatrix;
    max_gradient=1e-8,
)
    blk = Argos.backend(nlp)
    ncons = length.(blk.constraints.exprs)
    cnt = cumsum(ncons)

    # Powerflow
    con_scale[1:cnt[1]] .= 1e-0
    # Power generation
    con_scale[cnt[1]+1:cnt[2]] .= 1e0
    # Line flows
    con_scale[cnt[2]+1:cnt[3]] .= 1e0
    return
end

