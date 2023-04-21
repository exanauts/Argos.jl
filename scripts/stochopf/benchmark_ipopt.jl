
using MathOptInterface
using Ipopt
const MOI = MathOptInterface

include("common.jl")



function benchmark_ipopt(model, nscen; ntrials=3)
    blk = build_model(model, nscen; use_gpu=true)

    t_total, t_callbacks, t_linear_solver = (0.0, 0.0, 0.0)
    n_it = 0
    obj = 0

    optimizer = Ipopt.Optimizer()
    MOI.set(optimizer, MOI.RawOptimizerAttribute("print_level"), 5)
    MOI.set(optimizer, MOI.RawOptimizerAttribute("tol"), 1e-4)
    MOI.set(optimizer, MOI.RawOptimizerAttribute("max_iter"), 2)
    MOI.set(optimizer, MOI.RawOptimizerAttribute("linear_solver"), "ma27")
    # Warm-up
    solution = Argos.optimize!(optimizer, blk.nlp)

    MOI.empty!(optimizer)

    for _ in 1:ntrials
        MOI.empty!(optimizer)
        Argos.reset!(blk.nlp)
        MOI.set(optimizer, MOI.RawOptimizerAttribute("max_iter"), 250)
        time_solve = @elapsed begin
            solution = Argos.optimize!(optimizer, blk.nlp)
        end
        nlp_block = optimizer.nlp_data
        time_ad =
            nlp_block.evaluator.timers.update_time +
            nlp_block.evaluator.timers.obj_time +
            nlp_block.evaluator.timers.grad_time +
            nlp_block.evaluator.timers.cons_time +
            nlp_block.evaluator.timers.jacobian_time +
            nlp_block.evaluator.timers.hessian_time

        t_total += time_solve
        t_callbacks += time_ad
        t_linear_solver += time_solve - time_ad
        n_it += nlp_block.evaluator.counters.neval_hess
        obj += solution.minimum
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
    @assert CUDA.has_cuda()

    nexp = length(cases)
    results = zeros(nexp, 7)
    for (i, case) in enumerate(cases)
        @info "Case: $case"
        datafile = joinpath(DATA, case)
        model = ExaPF.PolarForm(datafile)
        nbus = PS.get(model, PS.NumberOfBuses())

        try
            r = benchmark_ipopt(model, nscens; ntrials=ntrials)
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
        output_file = joinpath(output_dir, "benchmark_ipopt_$(nscens).txt")
        writedlm(output_file, results)
    end
end

cases = [
    "case1354pegase.m",
    # "case2869pegase.m",
    # "case8387pegase.m",
    # "case_ACTIVSg500.m",
    "case_ACTIVSg2000.m",
    # "case_ACTIVSg10k.m",
]

main(cases, 8, 3, true)

