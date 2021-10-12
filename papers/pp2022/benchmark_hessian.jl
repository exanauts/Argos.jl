
using DelimitedFiles
using LinearAlgebra
using Printf
using Statistics
using SuiteSparse

using KernelAbstractions
using CUDAKernels

using ExaPF, ExaOpt

# Load GPU extension
include(joinpath(dirname(pathof(ExaOpt)), "..", "test", "cusolver.jl"))

OUTPUTDIR = joinpath(dirname(@__FILE__), "results")
SOURCE_DATA = joinpath(dirname(pathof(ExaOpt)), "..", "data")

function _instantiate_nlp(datafile, device, nbatches, line_constraints, pf_tol)
    # Instantiate problem
    pf = NewtonRaphson(tol=pf_tol)
    nlp = ExaOpt.ReducedSpaceEvaluator(
        datafile;
        device=device, nbatch_hessian=nbatches, line_constraints=line_constraints,
        powerflow_solver=pf,
    )
    u = ExaOpt.initial(nlp)
    g = similar(u)
    ExaOpt.update!(nlp, u)
    ExaOpt.gradient!(nlp, g, u)
    return (nlp, u)
end

"""
    Benchmark computation of objective's reduced Hessian ``∇²f``
    on instance `datafile`. Computation are deported on
    `device` (either `CPU()` or `CUDADevice()`).

"""
function benchmark_batched_hessian_objective(
    datafile, device;
    ntrials=50, line_constraints=false, pf_tol=1e-10,
)
    batches = [4, 8, 16, 32, 64, 128, 256, 512]

    timings = zeros(ntrials)
    results = zeros(length(batches) + 1, 5)

    # Reference is CPU (with batch=1 to fallback to default Hessian)
    nlp_ref, h_u = _instantiate_nlp(datafile, CPU(), 1, line_constraints, pf_tol)
    nu = ExaOpt.n_variables(nlp_ref)
    m = ExaOpt.n_constraints(nlp_ref)

    # Compute reference
    hess = zeros(nu, nu)
    for i in 1:ntrials
        t1 = @timed ExaOpt.hessian!(nlp_ref, hess, h_u)
        timings[i] = t1.time
    end
    results[1, 1] = mean(timings)
    results[1, 2] = median(timings)
    results[1, 3] = std(timings)
    results[1, 4] = length(timings)
    results[1, 5] = 1

    # Instantiate nlp on target device
    nlp, u = _instantiate_nlp(datafile, device, 2, line_constraints, pf_tol)
    model = ExaOpt.backend(nlp)
    # Instantiate Hessian
    hess = similar(u, nu, nu)

    # Get power flow's Jacobian
    J = ExaPF.powerflow_jacobian_device(model)
    func = line_constraints ? ExaPF.network_line_operations : ExaPF.network_operations

    for (id, nbatch) in enumerate(batches)
        (nbatch > nu) && break
        batch_ad = ExaOpt.BatchHessianLagrangian(model, func, J, nbatch, m)
        nlp.hesslag = batch_ad
        for i in 1:ntrials
            t1 = @timed ExaOpt.hessian!(nlp, hess, u)
            timings[i] = t1.time
        end
        results[id+1, 1] = mean(timings)
        results[id+1, 2] = median(timings)
        results[id+1, 3] = std(timings)
        results[id+1, 4] = length(timings)
        results[id+1, 5] = nbatch

        GC.gc(true)
        isa(device, GPU) && CUDA.reclaim()
    end
    return results
end

function launch_benchmark(; outputdir=OUTPUTDIR)
    results = Dict()

    for case in [
        "case118",
        "case300",
        "case1354pegase",
        "case2869pegase",
        "case9241pegase",
    ]
        @info 0 "Benchmark: " case
        datafile = joinpath(SOURCE_DATA, "$(case).m")

        res = benchmark_batched_hessian_objective(datafile, CUDADevice())

        results[case] = res
        output = joinpath(outputdir, "$(case)_batch_hessian.txt")
        writedlm(output, res)
        GC.gc(true)
        (CUDA.has_cuda_gpu()) && CUDA.reclaim()
    end

    return results
end

