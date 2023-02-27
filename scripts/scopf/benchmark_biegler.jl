
include("common.jl")

function _build_madnlp_gpu(blk::Argos.OPFModel)
    madnlp_options = Dict{Symbol, Any}()
    madnlp_options[:linear_solver] = LapackGPUSolver
    madnlp_options[:lapack_algorithm] = MadNLP.CHOLESKY
    madnlp_options[:dual_initialized] = true
    madnlp_options[:max_wall_time] = 1800.0
    madnlp_options[:max_iter] = 1000
    madnlp_options[:print_level] = MadNLP.DEBUG
    madnlp_options[:tol] = 1e-5
    opt_ipm, opt_linear, logger = MadNLP.load_options(; madnlp_options...)
    KKT = Argos.BieglerKKTSystem{Float64, CuVector{Int}, CuVector{Float64}, CuMatrix{Float64}}
    return MadNLP.MadNLPSolver{Float64, KKT}(blk, opt_ipm, opt_linear; logger=logger)
end

function benchmark_biegler(model, lines; ntrials=3)
    blk = build_model(model, lines; use_gpu=true)

    t_total, t_callbacks, t_linear_solver = (0.0, 0.0, 0.0)
    n_it = 0
    obj = 0

    solver = _build_madnlp_gpu(blk)
    # Warm-up
    MadNLP.solve!(solver; max_iter=1)

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

function main(cases, max_contingencies, ntrials, save_results)
    @assert CUDA.has_cuda()

    nexp = length(cases)
    results = zeros(nexp, 7)
    for (i, case) in enumerate(cases)
        @info "Case: $case"
        datafile = joinpath(DATA, case)
        model = ExaPF.PolarForm("../matpower/data/case_ACTIVSg2000.m")
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
            r = benchmark_biegler(model, lines; ntrials=ntrials)
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
        output_file = joinpath(output_dir, "benchmark_biegler_scopf_$(hash).txt")
        writedlm(output_file, results)
    end
end

cases = [
    "case_ACTIVSg2000.m",
]

main(cases, 1, 3, true)

