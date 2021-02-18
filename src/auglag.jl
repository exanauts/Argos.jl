
Base.@kwdef struct AugLagSolver <: AbstractExaOptimizer
    scaling::Bool = true
    max_iter::Int = 100
    max_inner_iter::Int = 1000
    ρ0::Float64 = 0.1
    ωtol::Float64 = 1e-5
    α0::Float64 = 1.0
    verbose::Int = 0
    inner_algo::Symbol = :tron
    lsq_lambda::Bool = false
    ε_primal::Float64 = 1e-8
    ε_dual::Float64 = 1e-8
end

function ExaPF.optimize!(
    algo::AugLagSolver,
    model::ExaPF.AbstractNLPEvaluator,
    u0::AbstractVector;
    options ...
)
    aug = ExaPF.AugLagEvaluator(model, u0; scale=algo.scaling, c₀=algo.ρ0)
    return ExaPF.optimize!(algo, aug, u0; options...)
end

# Augmented Lagrangian method
function ExaPF.optimize!(
    algo::AugLagSolver,
    aug::ExaPF.AugLagEvaluator,
    u0::AbstractVector;
    moi_optimizer=nothing
)
    nlp = aug.inner
    m = ExaPF.n_constraints(nlp)
    u♭, u♯ = ExaPF.bounds(nlp, ExaPF.Variables())

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

    ExaPF.update!(aug, uₖ)
    # Get gradient of Augmented Lagrangian
    ExaPF.gradient!(aug, grad, uₖ)
    feasible_direction!(wk, wk, uₖ, grad, 1.0, u♭, u♯)
    ε_primal = algo.ε_primal
    ε_dual = algo.ε_dual * (1.0 + norm(wk))

    ηk = 1.0 / (c0^0.1)

    # Init multiplier
    if algo.lsq_lambda
        copy!(aug.λ, ExaPF.estimate_multipliers(aug, uk))
    end

    local solution
    status = MOI.ITERATION_LIMIT

    verbose && log_header()
    for i_out in 1:algo.max_iter
        uₖ .= u_start
        # Inner iteration: projected gradient algorithm
        if algo.inner_algo == :ngpa
            solution = ngpa(aug, uₖ; α_bb=α0, α♯=α0, tol=ωtol, max_iter=algo.max_inner_iter)
        elseif algo.inner_algo == :projected_gradient
            solution = projected_gradient(aug, ųₖ; α0=α0, tol=ωtol)
        elseif algo.inner_algo == :tron
            solution = tron_solve(aug, uₖ;
                                  options=Dict("max_minor" => algo.max_inner_iter,
                                               "tron_code" => :Julia,
                                               "tol" => ωtol)
                                  )
        elseif algo.inner_algo == :MOI
            # Initiate optimizer
            optimizer = moi_optimizer()
            # Pass the problem to the ExaPF.MOIEvaluator
            moi_solution = ExaPF.optimize!(optimizer, aug, uₖ)
            MOI.empty!(optimizer)
            solution = (
                status=moi_solution.status,
                minimizer=moi_solution.minimizer,
                inf_du=1e8,
                iter=100,
            )
        end
        uₖ = solution.minimizer
        n_iter = solution.iter

        # Update information w.r.t. original evaluator
        obj = ExaPF.objective(nlp, uₖ)
        # Get gradient of Augmented Lagrangian
        ExaPF.gradient!(aug, grad, uₖ)
        feasible_direction!(wk, wk, uₖ, grad, 1.0, u♭, u♯)

        # Primal feasibility
        primal_feas = ExaPF.primal_infeasibility!(nlp, cons, uₖ)
        # Dual feasibility
        dual_feas = norm(wk, 2)

        # Log
        verbose && log_iter(i_out, obj, primal_feas, dual_feas, ηk, n_iter) # Log evolution
        push!(tracer, obj, primal_feas, dual_feas)

        if (dual_feas < ε_dual) && (primal_feas < ε_primal)
            status = MOI.OPTIMAL
            break
        end

        # Update starting point
        u_start .= uₖ
        # Update the penalties (see Nocedal & Wright, page 521)
        if norm(abs.(aug.cons), Inf) <= ηk
            ExaPF.update_multipliers!(aug)
            ηk = ηk / (aug.ρ^0.9)
        else
            ExaPF.update_penalty!(aug; η=10.0)
            ηk = 1.0 / (aug.ρ^0.1)
        end
    end

    if verbose
        println("Number of objective function evaluations             = ", aug.counter.objective)
        println("Number of objective gradient evaluations             = ", aug.counter.gradient)
    end

    solution = (
        status=status,
        minimum=obj,
        minimizer=uₖ,
        trace=tracer,
    )

    return solution
end

