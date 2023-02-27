using Dates
using DelimitedFiles
using Graphs
using LazyArtifacts
using Printf
using Random

using NLPModels
using Argos
using ExaPF
using MadNLP

# HSL
using MadNLPHSL

# GPU
using KernelAbstractions
using CUDA
using CUDAKernels
using ArgosCUDA
using MadNLPGPU

const PS = ExaPF.PowerSystem

const DATA = joinpath(artifact"ExaData", "ExaData")
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

function build_model(model, lines; use_gpu=false)
    contingencies = [ExaPF.LineContingency(l) for l in lines]
    nblks = length(contingencies) + 1
    if use_gpu
        model_gpu = PolarForm(model, CUDADevice())
        nlp = Argos.StochEvaluator(model_gpu, nblks; contingencies=contingencies)
        blk = Argos.OPFModel(Argos.bridge(nlp))
    else
        nlp = Argos.StochEvaluator(model, nblks; contingencies=contingencies)
        blk = Argos.OPFModel(nlp)
    end
    init_model!(blk)
    return blk
end

# Custom scaling
function MadNLP.scale_objective(nlp::Argos.OPFModel, grad::AbstractVector; max_gradient=1e-8)
    return 1e-3
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
    con_scale[cnt[1]+1:cnt[2]] .= 1e-2
    # Line flows
    con_scale[cnt[2]+1:cnt[3]] .= 1e-2
    return
end

