
include("common.jl")

function _build_madnlp_cpu(blk::Argos.OPFModel)
    return MadNLP.MadNLPSolver(
        blk;
        linear_solver=Ma27Solver,
        dual_initialized=true,
        max_wall_time=1800.0,
        max_iter=1000,
        print_level=MadNLP.DEBUG,
        tol=1e-5,
    )
end

function benchmark_extensive(model, lines; ntrials=3, gpu_ad=false)
    blk = build_model(model, lines; use_gpu=gpu_ad)

    t_total, t_callbacks, t_linear_solver = (0.0, 0.0, 0.0)
    n_it = 0
    obj = 0

    solver = _build_madnlp_cpu(blk)
    # Warm-up
    MadNLP.solve!(solver; max_iter=1)

    for _ in 1:ntrials
        solver = _build_madnlp_cpu(blk)
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

function main(cases, max_contingencies, ntrials, save_results)
    gpu_ad = (CUDA.has_cuda()) ? true : false

    nexp = length(cases)
    results = zeros(nexp, 7)
    for (i, case) in enumerate(cases)
        @info "Case: $case"
        datafile = joinpath(DATA, case)
        model = ExaPF.PolarForm(datafile)
        nbus = PS.get(model, PS.NumberOfBuses())

        instance = split(case, ".")[1]
        lines = readdlm(joinpath(SCENARIOS, "$(instance).Ctgs"), ',', Int)[:]
        lines = screen_contingencies(model.network, lines)
        if max_contingencies >= 1 && length(lines) > max_contingencies
            lines = lines[1:max_contingencies]
        end
        nscens = length(lines)
        println("Num contingencies: ", nscens)

        try
            r = benchmark_extensive(model, lines; ntrials=ntrials, gpu_ad=gpu_ad)
            results[i, :] .= (nbus, nscens, r.iters, r.obj, r.total, r.callbacks, r.linear_solver)
        catch
            println("fail to solve problem $case.")
        end
        refresh_memory()
    end

    if save_results
        output_dir = joinpath(dirname(@__FILE__), RESULTS_DIR)
        if !isdir(output_dir)
            mkdir(output_dir)
        end
        time_now = now()
        hash = "$(today())-$(hour(time_now))h$(minute(time_now))"
        output_file = joinpath(output_dir, "benchmark_ma27_scopf_$(gpu_ad)_$(hash).txt")
        writedlm(output_file, results)
    end
end

cases = [
    "case_ACTIVSg2000.m",
]

main(cases, 1, 1, true)

