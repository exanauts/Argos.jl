
include("common.jl")

function build_solver(blk::Argos.OPFModel)
    return MadNLP.MadNLPSolver(
        blk;
        dual_initialized=true,
        linear_solver=Ma27Solver,
        max_wall_time=1800.0,
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

function main(cases, nscens, ntrials, save_results)
    gpu_ad = (CUDA.has_cuda()) ? true : false

    nexp = length(cases)
    results = zeros(nexp, 7)
    for (i, case) in enumerate(cases)
        @info "Case: $case"
        datafile = joinpath(DATA, case)
        model = ExaPF.PolarForm(datafile)
        nbus = PS.get(model, PS.NumberOfBuses())

        r = benchmark_extensive(model, nscens; ntrials=ntrials, gpu_ad=gpu_ad)
        results[i, :] .= (nbus, nscens, r.iters, r.obj, r.total, r.callbacks, r.linear_solver)
        refresh_memory()
    end

    if save_results
        output_dir = joinpath(dirname(@__FILE__), RESULTS_DIR)
        if !isdir(output_dir)
            mkdir(output_dir)
        end
        output_file = joinpath(output_dir, "benchmark_ma27_$(nscens).txt")
        writedlm(output_file, results)
    end
end

cases = [
    "case1354pegase.m",
    "case2869pegase.m",
    "case8387pegase.m",
    "case_ACTIVSg500.m",
    "case_ACTIVSg2000.m",
    "case_ACTIVSg10k.m",
]

main(cases, 8, 3, true)

