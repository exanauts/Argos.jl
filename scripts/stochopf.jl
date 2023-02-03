using DelimitedFiles
using KernelAbstractions
using LazyArtifacts
using Random

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

function build_solver(blk::Argos.OPFModel)
    return MadNLP.MadNLPSolver(
        blk;
        dual_initialized=true,
        linear_solver=Ma27Solver,
        max_iter=250,
        print_level=MadNLP.DEBUG,
        tol=1e-5,
    )
end

function benchmark_extensive(model, nscen; ntrials=3, gpu_ad=false)
    blk = build_model(model, nscen; use_gpu=gpu_ad)

    t_total, t_callbacks, t_linear_solver = (0.0, 0.0, 0.0)
    n_it = 0
    obj = 0

    solver = build_solver(blk)
    # Warm-up
    MadNLP.solve!(solver; max_iter=2)

    for _ in 1:ntrials
        solver = build_solver(blk)
        MadNLP.solve!(solver)
        t_total += solver.cnt.total_time
        t_callbacks += solver.cnt.eval_function_time
        t_linear_solver += solver.cnt.linear_solver_time
        n_it += solver.cnt.k
        obj += solver.obj_val
    end
    return (
        iters = n_it / ntrials,
        obj = obj / ntrials,
        total = t_total / ntrials,
        callbacks = t_callbacks / ntrials,
        linear_solver = t_linear_solver / ntrials,
    )
end

function _build_madnlp_gpu(blk::Argos.OPFModel)
    madnlp_options = Dict{Symbol, Any}()
    madnlp_options[:linear_solver] = LapackGPUSolver
    madnlp_options[:lapack_algorithm] = MadNLP.CHOLESKY
    madnlp_options[:dual_initialized] = true
    madnlp_options[:max_iter] = 250
    madnlp_options[:print_level] = MadNLP.DEBUG
    madnlp_options[:tol] = 1e-5
    opt_ipm, opt_linear, logger = MadNLP.load_options(; madnlp_options...)
    KKT = Argos.BieglerKKTSystem{Float64, CuVector{Int}, CuVector{Float64}, CuMatrix{Float64}}
    return MadNLP.MadNLPSolver{Float64, KKT}(blk, opt_ipm, opt_linear; logger=logger)
end

function benchmark_biegler(model, nscen; ntrials=3)
    blk = build_model(model, nscen; use_gpu=true)

    t_total, t_callbacks, t_linear_solver = (0.0, 0.0, 0.0)
    n_it = 0
    obj = 0

    solver = _build_madnlp_gpu(blk)
    # Warm-up
    MadNLP.solve!(solver; max_iter=2)

    for _ in 1:ntrials
        solver = _build_madnlp_gpu(blk)
        MadNLP.solve!(solver)
        t_total += solver.cnt.total_time
        t_callbacks += solver.cnt.eval_function_time
        t_linear_solver += solver.cnt.linear_solver_time
        n_it += solver.cnt.k
        obj += solver.obj_val
    end
    return (
        iters = n_it / ntrials,
        obj = obj / ntrials,
        total = t_total / ntrials,
        callbacks = t_callbacks / ntrials,
        linear_solver = t_linear_solver / ntrials,
    )
end
