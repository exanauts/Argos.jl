include("common.jl")
include("config.jl")
include("screening.jl")

function build_madnlp(
    blk::Argos.OPFModel,
    ::Argos.FullSpace;
    max_iter=250,
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
    max_iter=250,
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

function benchmark(model, lines, kkt; use_gpu=false, ntrials=3, options...)
    blk = build_scopf_model(model, lines; use_gpu=use_gpu)

    ## Warm-up
    solver = build_madnlp(blk, kkt; max_iter=1)
    MadNLP.solve!(solver)

    ## Benchmark
    t_total, t_callbacks, t_linear_solver = (0.0, 0.0, 0.0)
    n_it, obj = 0, 0.0
    for _ in 1:ntrials
        Argos.reset!(Argos.backend(blk))
        # Solve
        solver = build_madnlp(blk, kkt; options...)
        MadNLP.solve!(solver)
        # Save results
        t_total += solver.cnt.total_time
        t_callbacks += solver.cnt.eval_function_time
        t_linear_solver += solver.cnt.linear_solver_time
        n_it += solver.cnt.k
        obj += solver.obj_val
        # Clean memory
        use_gpu && refresh_memory()
    end

    return (
        iters = n_it / ntrials,
        obj = obj / ntrials,
        total = t_total / ntrials,
        callbacks = t_callbacks / ntrials,
        linear_solver = t_linear_solver / ntrials,
    )
end

function main(cases, kkt, max_contingencies, ntrials, save_results; use_gpu=false, src_contingencies=:exadata, options...)
    # Setup
    dev = use_gpu ? :cuda : :cpu
    form = isa(kkt, Argos.BieglerReduction) ? :biegler : :full

    nexp = length(cases)
    results = zeros(nexp, 7)

    for (i, case) in enumerate(cases)
        @info "Case: $case"
        datafile = joinpath(DATA, case)
        model = ExaPF.PolarForm(datafile)
        nbus = PS.get(model, PS.NumberOfBuses())
        nlines = PS.get(model, PS.NumberOfLines())

        if src_contingencies == :exadata
            instance = split(case, ".")[1]
            lines = readdlm(joinpath(SCENARIOS, "$(instance)_onehour_60.Ctgs"), ',', Int)[:]
            lines = filter_islanding(model.network, lines)
        elseif src_contingencies == :generated
            all_lines = collect(1:nlines)
            # Basic screening to remove islanding
            lines = filter_islanding(model.network, all_lines)
        else
            error("Only :exadata and :generated are supported for contingencies")
        end

        if max_contingencies >= 1 && length(lines) > max_contingencies
            lines = lines[1:max_contingencies]
        end
        nscens = length(lines)
        println("Num contingencies: ", nscens)

        try
            r = benchmark(model, lines, kkt; ntrials=ntrials, use_gpu=use_gpu, options...)
            results[i, :] .= (nbus, nscens, r.iters, r.obj, r.total, r.callbacks, r.linear_solver)
        catch err
            println("Error when solving problem $case: $(err)")
        end
    end

    if save_results
        output_dir = joinpath(dirname(@__FILE__), RESULTS_DIR)
        if !isdir(output_dir)
            mkdir(output_dir)
        end
        output_file = joinpath(output_dir, "benchmark_opf_$(form)_$(dev).txt")
        writedlm(output_file, results)
    end
    return results
end

main(
    cases,
    kkt,
    max_contingencies,
    ntrials,
    save_results;
    print_level=print_level,
    src_contingencies=src_contingencies,
    linear_solver=linear_solver,
    use_gpu=use_gpu,
)

