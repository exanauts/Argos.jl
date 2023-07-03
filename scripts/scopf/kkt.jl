include("common.jl")
include("config.jl")
include("screening.jl")

function analyse_memory(A::CUSPARSE.CuSparseMatrixCSR)
    tot_mem = 0
    tot_mem += sizeof(A.nzVal)
    tot_mem += sizeof(A.rowPtr)
    tot_mem += sizeof(A.colVal)
    return tot_mem
end

function analyse_memory(F::ArgosCUDA.CUSOLVERRF.RFLU)
    tot_mem = 0
    tot_mem += analyse_memory(F.M)
    tot_mem += analyse_memory(F.P)
    tot_mem += analyse_memory(F.Q)
    tot_mem += sizeof(F.T)
    tot_mem += sizeof(F.dsm.buffer)
    tot_mem += sizeof(F.tsm.buffer)
    tot_mem += sizeof(F.tsv.buffer)
    return tot_mem
end

function analyse_memory(kkt::Argos.BieglerKKTSystem)
    tot_mem = 0
    # Buffers
    tot_mem += sizeof(kkt._wj1)
    tot_mem += sizeof(kkt._wx1)
    tot_mem += sizeof(kkt._wx2)
    tot_mem += sizeof(kkt._wxu1)
    tot_mem += sizeof(kkt._wxu2)
    tot_mem += sizeof(kkt._wxu3)

    # Reduced matrix
    tot_mem += sizeof(kkt.aug_com)

    # Sparse transformation
    tot_mem += sizeof(kkt.h_V)
    tot_mem += sizeof(kkt.j_V)
    tot_mem += analyse_memory(kkt.Gx)
    tot_mem += analyse_memory(kkt.Gu)
    tot_mem += analyse_memory(kkt.A)
    tot_mem += sizeof(kkt.j_V)

    # IPM
    tot_mem += sizeof(kkt.pr_diag)
    tot_mem += sizeof(kkt.du_diag)
    tot_mem += sizeof(kkt.con_scale)
    tot_mem += sizeof(kkt.jacobian_scaling)

    # Condensed
    tot_mem += sizeof(kkt.K.Σ)
    tot_mem += sizeof(kkt.K.transperm)
    tot_mem += analyse_memory(kkt.K.JtJ)
    tot_mem += analyse_memory(kkt.K.Jt)
    tot_mem += analyse_memory(kkt.K.W)

    #
    tot_mem += analyse_memory(kkt.G_fac)
    tot_mem += sizeof(kkt.reduction.z)
    tot_mem += sizeof(kkt.reduction.ψ)
    tot_mem += sizeof(kkt.reduction.tangents)

    return tot_mem
end

function benchmark_kkt(model, lines, kkt; use_gpu=false, ntrials=3, options...)
    use_gpu && refresh_memory()
    blk = build_scopf_model(model, lines; use_gpu=use_gpu)

    ## Warm-up
    solver = build_madnlp(blk, kkt; max_iter=1)
    MadNLP.solve!(solver)

    ## Benchmark
    t_build, t_factorization, t_backsolve = (0.0, 0.0, 0.0)
    delta_err = 0.0
    n_it, obj = 0, 0.0
    for _ in 1:ntrials
        t_build += CUDA.@elapsed begin
            MadNLP.build_kkt!(solver.kkt)
        end
        t_factorization += CUDA.@elapsed begin
            MadNLP.factorize!(solver.linear_solver)
        end
        t_backsolve += CUDA.@elapsed begin
            MadNLP.solve_refine_wrapper!(solver, solver.d, solver.p)
        end

        dsol = MadNLP.primal_dual(solver.d)
        n = length(dsol)
        psol = zeros(n)

        mul!(psol, solver.kkt, dsol)

        delta_err += norm(psol .- MadNLP.primal_dual(solver.p), Inf)
    end

    memory_usage = if isa(kkt, Argos.BieglerReduction)
        analyse_memory(solver.kkt)
    else
        0
    end
    println(memory_usage)

    return (
        build=t_build / ntrials,
        factorization=t_factorization / ntrials,
        backsolve=t_backsolve / ntrials,
        accuracy=delta_err / ntrials,
        memory=memory_usage,
    )
end

function launch_kkt(cases, case_contingencies, kkt, ntrials, save_results; use_gpu=false, src_contingencies=:exadata, options...)
    # Setup
    dev = use_gpu ? :cuda : :cpu
    form = isa(kkt, Argos.BieglerReduction) ? :biegler : :full

    nexp = sum(length.(case_contingencies))
    results = zeros(nexp, 7)

    i = 0
    for (case, contingencies) in zip(cases, case_contingencies)
        datafile = joinpath(DATA, case)
        model = ExaPF.PolarForm(datafile)
        nbus = PS.get(model, PS.NumberOfBuses())
        nlines = PS.get(model, PS.NumberOfLines())
        if src_contingencies == :exadata
            instance = split(case, ".")[1]
            all_lines = readdlm(joinpath(SCENARIOS, "$(instance)_onehour_60.Ctgs"), ',', Int)[:]
            all_lines = filter_islanding(model.network, all_lines)
        elseif src_contingencies == :generated
            all_lines = collect(1:nlines)
            # Basic screening to remove islanding
            all_lines = filter_islanding(model.network, all_lines)
        else
            error("Only :exadata and :generated are supported for contingencies")
        end

        for ncont in contingencies
            i += 1
            @info "Case: $case N: $ncont"
            lines = all_lines[1:ncont]
            r = benchmark_kkt(model, lines, kkt; ntrials=ntrials, use_gpu=use_gpu, options...)
            results[i, :] .= (nbus, ncont, r.build, r.factorization, r.backsolve, r.accuracy, r.memory / 1024^2)
        end
    end

    if save_results
        output_dir = joinpath(dirname(@__FILE__), RESULTS_DIR)
        if !isdir(output_dir)
            mkdir(output_dir)
        end
        output_file = joinpath(output_dir, "benchmark_kkt_$(form)_$(dev).txt")
        writedlm(output_file, results)
    end
    return results
end

launch_kkt(
    cases,
    case_contingencies,
    kkt,
    ntrials,
    save_results;
    print_level=print_level,
    src_contingencies=src_contingencies,
    linear_solver=linear_solver,
    use_gpu=use_gpu,
)

