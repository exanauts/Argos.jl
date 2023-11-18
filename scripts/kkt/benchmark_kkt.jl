include("common.jl")

#=
    CONFIG
=#

# Number of trial runs to estimate running time.
ntrials = 3
# Save results on disk?
save_results = true
# Should we use the GPU to evaluate the derivatives?
use_gpu = true
# Verbose level
verbose = true
print_level = if verbose
    MadNLP.DEBUG
else
    MadNLP.ERROR
end

# OPF instances
cases = [
    "case118.m",
    "case1354pegase.m",
    "case2869pegase.m",
    "case9241pegase.m",
]


function benchmark_kkt(model, kkt; use_gpu=false, ntrials=3, options...)
    use_gpu && refresh_memory()
    blk = build_opf_model(model; use_gpu=use_gpu)

    ## Warm-up
    solver = build_madnlp(blk, kkt; max_iter=1, options...)
    MadNLP.solve!(solver)

    ## Benchmark
    t_build, t_factorization, t_backsolve = (0.0, 0.0, 0.0)
    delta_err = 0.0
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

    return (
        build=t_build / ntrials,
        factorization=t_factorization / ntrials,
        backsolve=t_backsolve / ntrials,
        accuracy=delta_err / ntrials,
    )
end

function benchmark_kkt(cases, kkt, ntrials, save_results; use_gpu=false, options...)
    # Setup
    dev = use_gpu ? :cuda : :cpu
    form = isa(kkt, Argos.BieglerReduction) ? :biegler : :full

    nexp = length(cases)
    results = zeros(nexp, 5)

    i = 0
    for case in cases
        i += 1
        datafile = joinpath(DATA, case)
        model = ExaPF.PolarForm(datafile)
        nbus = PS.get(model, PS.NumberOfBuses())

        r = benchmark_kkt(model, kkt; ntrials=ntrials, use_gpu=use_gpu, options...)
        results[i, :] .= (nbus, r.build, r.factorization, r.backsolve, r.accuracy)
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

#=
    Benchmark using ma27 as a reference.
=#
benchmark_kkt(
    cases,
    Argos.FullSpace(),
    ntrials,
    save_results;
    print_level=print_level,
    linear_solver=Ma27Solver,
    use_gpu=use_gpu,
)


#=
    Benchmark Biegler's reduction.
=#
benchmark_kkt(
    cases,
    Argos.BieglerReduction(),
    ntrials,
    save_results;
    print_level=print_level,
    linear_solver=LapackGPUSolver,
    use_gpu=use_gpu,
)

