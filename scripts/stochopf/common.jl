using DelimitedFiles
using KernelAbstractions
using LazyArtifacts
using Random

using NLPModels
using Argos
using ExaPF
using MadNLP

# HSL
using MadNLPHSL

# GPU
using CUDA
using CUDAKernels
using ArgosCUDA
using MadNLPGPU

const PS = ExaPF.PowerSystem

const DATA = "/home/fpacaud/dev/matpower/data/"
const DEMANDS = joinpath(artifact"ExaData", "ExaData", "mp_demand")
RESULTS_DIR = "results"

if CUDA.has_cuda()
    CUDA.allowscalar(false)
end

function refresh_memory()
    GC.gc(true)
    CUDA.has_cuda() && CUDA.reclaim()
    return
end

function import_loads(casename)
    pload = readdlm(joinpath(DEMANDS, "$(casename)_onehour_60.Pd")) ./ 100
    qload = readdlm(joinpath(DEMANDS, "$(casename)_onehour_60.Qd")) ./ 100
    return pload, qload
end

function generate_loads(model, nscen, magnitude)
    nbus = get(model, ExaPF.PS.NumberOfBuses())
    stack = ExaPF.NetworkStack(model)
    pload_det = stack.pload |> Array
    qload_det = stack.qload |> Array

    has_load = (pload_det .> 0)

    Random.seed!(1)
    pload = magnitude .* (randn(nbus, nscen) .* has_load) .+ pload_det
    qload = magnitude .* (randn(nbus, nscen) .* has_load) .+ qload_det
    return pload, qload
end

function build_model(model, nscen; use_gpu=false)
    pload, qload = generate_loads(model, nscen, 0.0)
    if use_gpu
        model_gpu = PolarForm(model, CUDADevice())
        nlp = Argos.StochEvaluator(model_gpu, pload, qload)
        blk = Argos.OPFModel(Argos.bridge(nlp))
    else
        nlp = Argos.StochEvaluator(model, pload, qload)
        blk = Argos.OPFModel(nlp)
    end

    # Instantiate
    x0 = NLPModels.get_x0(blk)
    nnzj = NLPModels.get_nnzj(blk)
    jac = zeros(nnzj)
    NLPModels.jac_coord!(blk, x0, jac)
    return blk
end

