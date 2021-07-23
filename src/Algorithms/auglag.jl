
Base.@kwdef struct AugLagSolver <: AbstractExaOptimizer
    scaling::Bool = true
    max_iter::Int = 100
    max_inner_iter::Int = 1000
    ρ0::Float64 = 0.1
    rate::Float64 = 10.0
    ωtol::Float64 = 1e-5
    α0::Float64 = 1.0
    verbose::Int = 0
    inner_algo::Symbol = :tron
    lsq_lambda::Bool = false
    ε_primal::Float64 = 1e-8
    ε_dual::Float64 = 1e-8
end

function optimize!(
    algo::AugLagSolver,
    model::AbstractNLPEvaluator,
    u0::AbstractVector;
    options ...
)
    aug = ExaOpt.AugLagEvaluator(model, u0; scale=algo.scaling, c₀=algo.ρ0)
    return optimize!(algo, aug, u0; options...)
end

# Augmented Lagrangian method
function optimize!(
    algo::AugLagSolver,
    aug::AugLagEvaluator,
    u0::AbstractVector;
    moi_optimizer=nothing
)
    nlp = aug.inner
    m = n_constraints(nlp)
    u♭, u♯ = bounds(nlp, Variables())

    # Initialize arrays
    uₖ        = copy(u0)
    u_start   = copy(u0)
    wk        = copy(u0)
    u_prev    = copy(u0)
    grad      = similar(u0) ; fill!(grad, 0)
    ut        = similar(u0) ; fill!(ut, 0)
    cons      = similar(u0, m) ; fill!(cons, 0)

    obj = Inf
    norm_grad = Inf

    tracer = Tracer()

    c0 = algo.ρ0
    ωtol = algo.ωtol
    α0 = algo.α0
    verbose = (algo.verbose > 0)

    # Initialization (aka iteration 0)
    update!(aug, uₖ)
    # Get gradient of Augmented Lagrangian
    gradient!(aug, grad, uₖ)
    feasible_direction!(wk, wk, uₖ, grad, 1.0, u♭, u♯)

    ε_primal = algo.ε_primal
    ε_dual = algo.ε_dual * (1.0 + norm(wk))

    ηk = 1.0 / (c0^0.1)

    # Init multiplier
    if algo.lsq_lambda
        copy!(aug.λ, estimate_multipliers(aug, uₖ))
    end

    if verbose
        name = if algo.inner_algo == :MOI
            MOI.get(moi_optimizer(), MOI.SolverName())
        else
            algo.inner_algo
        end
        println("AugLag algorithm, running with $(name)\n")

        println("Total number of variables............................:      ", n_variables(nlp))
        println("Total number of constraints..........................:      ", n_constraints(nlp))
        println()

        log_header()
        # O-th iteration
        obj = objective(nlp, uₖ)
        primal_feas = primal_infeasibility!(nlp, cons, uₖ)
        dual_feas = norm(wk, 2)
        log_iter(0, obj, primal_feas, dual_feas, ηk, aug.ρ, 0)
    end

    local solution
    status = MOI.ITERATION_LIMIT
    mul = copy(aug.λ)

    tic = time()
    for i_out in 1:algo.max_iter
        uₖ .= u_start
        # Inner iteration: projected gradient algorithm
        if algo.inner_algo == :ngpa
            solution = ngpa(aug, uₖ; α_bb=α0, α♯=α0, tol=ωtol, max_iter=algo.max_inner_iter)
        elseif algo.inner_algo == :projected_gradient
            solution = projected_gradient(aug, ųₖ; α0=α0, tol=ωtol)
        elseif algo.inner_algo == :tron
            solution = tron_solve(
                aug, uₖ;
                options=Dict(
                    "max_minor" => algo.max_inner_iter,
                    "tron_code" => :Julia,
                    "tol" => ωtol
                )
            )
        elseif algo.inner_algo == :MOI
            # Initiate optimizer
            optimizer = moi_optimizer()
            # Pass the problem to the MOIEvaluator
            n_iter = aug.counter.gradient
            moi_solution = optimize!(optimizer, aug, uₖ)
            MOI.empty!(optimizer)
            solution = (
                status=moi_solution.status,
                iter=aug.counter.gradient - n_iter,
                minimizer=moi_solution.minimizer,
            )
        end
        uₖ = solution.minimizer
        n_iter = solution.iter

        # Update information w.r.t. original evaluator
        obj = objective(nlp, uₖ)
        # Get gradient of Augmented Lagrangian
        gradient!(aug, grad, uₖ)
        feasible_direction!(wk, wk, uₖ, grad, 1.0, u♭, u♯)

        # Primal feasibility
        primal_feas = primal_infeasibility!(nlp, cons, uₖ)
        # Dual feasibility
        dual_feas = norm(wk, 2)

        # Log
        verbose && log_iter(i_out, obj, primal_feas, dual_feas, ηk, aug.ρ, n_iter) # Log evolution
        push!(tracer, obj, primal_feas, dual_feas)

        if (dual_feas < ε_dual) && (primal_feas < ε_primal)
            status = MOI.OPTIMAL
            break
        end

        # Update starting point
        u_start .= uₖ
        # Update the penalties (see Nocedal & Wright, page 521)
        if norm(abs.(aug.cons), Inf) <= ηk
            update_multipliers!(aug)
            mul = hcat(mul, aug.λ)
            ηk = ηk / (aug.ρ^0.9)
        else
            update_penalty!(aug; η=algo.rate)
            ηk = 1.0 / (aug.ρ^0.1)
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

