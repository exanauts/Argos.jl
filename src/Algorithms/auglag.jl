
Base.@kwdef struct AugLagOptions
    scaling::Bool = true
    max_iter::Int = 100
    max_inner_iter::Int = 1000
    rate::Float64 = 10.0
    ωtol::Float64 = 1.0
    ωtol_min::Float64 = 1.0e-5
    α0::Float64 = 1.0
    verbose::Int = 0
    inner_algo::Symbol = :tron
    lsq_lambda::Bool = false
    ε_primal::Float64 = 1e-8
    ε_dual::Float64 = 1e-8
end

struct AuglagSolver{InnerOptimizer} <: AbstractExaOptimizer
    optimizer::InnerOptimizer
    options::AugLagOptions
end

function AuglagSolver(optimizer; options...)
    return AuglagSolver(optimizer, AugLagOptions(; options...))
end

# Solve subproblem with MadNLP
function solve_subproblem!(
    algo::AuglagSolver{<:MadNLP.MadNLPSolver}, aug::AugLagEvaluator, uₖ;
    tol=-1, niter=1
)
    n_iter = aug.counter.hessian
    # Init primal variable
    x0 = NLPModels.get_x0(algo.optimizer.nlp)
    copyto!(x0, uₖ)
    algo.optimizer.nlp.hash_x[] = UInt(0) # reset hash to do at least one iteration
    # Set initial mu if resolve
    if algo.optimizer.status != MadNLP.INITIAL
        if niter < 6
            algo.optimizer.opt.mu_init = 1e-4
        elseif niter < 10
            algo.optimizer.opt.mu_init = 1e-6
        else
            algo.optimizer.opt.mu_init = 1e-8
        end
    end
    # Optimize with IPM
    res = MadNLP.solve!(algo.optimizer)
    return (
        status=MadNLP._STATUS_CODES[res.status],
        iter=aug.counter.hessian - n_iter,
        minimizer=res.solution,
    )
end

get_name(solver::AuglagSolver{<:MOI.AbstractOptimizer}) = MOI.get(solver.optimizer, MOI.SolverName())
get_name(solver::AuglagSolver{<:MadNLP.MadNLPSolver}) = "MadNLP+"

#=
    CORE ALGORITHM
=#

# Augmented Lagrangian method
function optimize!(
    algo::AuglagSolver,
    aug::AugLagEvaluator,
    u0::AbstractVector;
)
    opt = algo.options
    nlp = aug.inner
    m = n_constraints(nlp)
    u♭, u♯ = bounds(nlp, Variables())

    # Initialize arrays
    uₖ        = copy(u0)
    dₖ        = copy(u0)
    u_prev    = copy(u0)
    grad      = similar(u0) ; fill!(grad, 0)
    ut        = similar(u0) ; fill!(ut, 0)
    cons      = similar(u0, m) ; fill!(cons, 0)

    obj = Inf
    norm_grad = Inf

    tracer = Tracer()

    ρ0 = aug.ρ
    ωtol = opt.ωtol
    α0 = opt.α0
    verbose = (opt.verbose > 0)

    # Initialization (aka iteration 0)
    update!(aug, uₖ)
    # Get gradient of Augmented Lagrangian
    gradient!(aug, grad, uₖ)
    feasible_direction!(dₖ, uₖ, grad, 1.0, u♭, u♯)

    ε_primal = opt.ε_primal
    ε_dual = opt.ε_dual * max(1.0, norm(dₖ))
    !isnothing(aug.tracker) && (aug.tracker.ext[:scale_sd] = norm(dₖ))

    ηk = 1.0 / (ρ0^0.1)

    # Init multiplier
    if opt.lsq_lambda
        copy!(aug.λ, estimate_multipliers(aug, uₖ))
    end

    if verbose
        name = get_name(algo)
        println("AugLag algorithm, running with $(name)\n")

        println("Total number of variables............................:      ", n_variables(nlp))
        println("Total number of constraints..........................:      ", n_constraints(nlp))
        println()

        log_header()
        # O-th iteration
        obj = objective(nlp, uₖ)
        primal_feas = primal_infeasibility!(nlp, cons, uₖ)
        dual_feas = norm(dₖ, 2)
        log_iter(0, obj, primal_feas, dual_feas, ηk, aug.ρ, 0)
    end

    local solution
    status = MOI.ITERATION_LIMIT
    mul = copy(aug.λ)

    tic = time()
    for i_out in 1:opt.max_iter

        # Solve inner problem
        solution = solve_subproblem!(algo, aug, uₖ; niter=i_out)

        if (solution.status != MOI.OPTIMAL) &&
           (solution.status != MOI.LOCALLY_SOLVED)  &&
           (solution.status != MOI.SLOW_PROGRESS) &&
           (solution.status != MOI.ITERATION_LIMIT)
            println("[AugLag] Fail to solve inner subproblem. Status: $(solution.status). Exiting.")
            status = MOI.NUMERICAL_ERROR
            break
        end

        copyto!(uₖ, solution.minimizer)
        n_iter = solution.iter

        # Update information w.r.t. original evaluator
        obj = objective(nlp, uₖ)
        # Get gradient of Augmented Lagrangian
        gradient!(aug, grad, uₖ)
        # d = P[u - grad] - u
        feasible_direction!(dₖ, uₖ, grad, 1.0, u♭, u♯)

        # Primal feasibility
        primal_feas = norm(aug.cons, Inf)
        # Dual feasibility
        dual_feas = norm(dₖ, 2)

        # Update the penalties (see Nocedal & Wright, page 521)
        if primal_feas <= ηk
            update_multipliers!(aug)
            mul = hcat(mul, aug.λ)
            ηk = ηk / (aug.ρ^0.9)
        else
            update_penalty!(aug; η=opt.rate)
            ηk = 1.0 / (aug.ρ^0.1)
        end

        # Log
        verbose && log_iter(i_out, obj, primal_feas, dual_feas, ηk, aug.ρ, n_iter) # Log evolution
        push!(tracer, obj, primal_feas, dual_feas)

        if (dual_feas < ε_dual) && (primal_feas < ε_primal)
            status = MOI.OPTIMAL
            break
        end
    end
    toc = time() - tic

    if verbose
        println()
        println("Number of iterations....: ", length(tracer.objective))
        println("Number of objective function evaluations             = ", aug.counter.objective)
        println("Number of objective gradient evaluations             = ", aug.counter.gradient)
        println("Number of Lagrangian Hessian evaluations             = ", aug.counter.hessian)
        @printf("Total CPU time                                       = %.3f\n", toc)
        println()
    end

    solution = (
        status=status,
        minimum=obj,
        minimizer=uₖ,
        trace=tracer,
        multipliers=mul,
    )

    return solution
end

