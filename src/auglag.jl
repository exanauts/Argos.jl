
Base.@kwdef struct AugLagSolver <: AbstractExaOptimizer
    scaling::Bool = true
    max_iter::Int = 100
    max_inner_iter::Int = 1000
    ρ0::Float64 = 0.1
    ωtol::Float64 = 1e-5
    α0::Float64 = 1.0
    verbose::Int = 0
    inner_algo::Symbol = :tron
    lsq_lambda::Bool = true
end

function ExaPF.optimize!(
    algo::AugLagSolver,
    model::ExaPF.AbstractNLPEvaluator,
    u0::AbstractVector;
    options ...
)
    aug = ExaPF.AugLagEvaluator(model, u0; scale=algo.scaling)
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

    # Initialize arrays
    uk        = copy(u0)
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

    ηk = 1.0 / (c0^0.1)

    # Init multiplier
    if algo.lsq_lambda
        copy!(aug.λ, ExaPF.estimate_multipliers(aug, uk))
    end

    local solution

    verbose && log_header()
    for i_out in 1:algo.max_iter
        uk .= u_start
        # Inner iteration: projected gradient algorithm
        if algo.inner_algo == :ngpa
            solution = ngpa(aug, uk; α_bb=α0, α♯=α0, tol=ωtol, max_iter=algo.max_inner_iter)
        elseif algo.inner_algo == :projected_gradient
            solution = projected_gradient(aug, uk; α0=α0, tol=ωtol)
        elseif algo.inner_algo == :tron
            solution = tron_solve(aug, uk;
                                  options=Dict("max_minor" => algo.max_inner_iter,
                                               "tron_code" => :Julia,
                                               "tol" => ωtol)
                                  )
        elseif algo.inner_algo == :MOI
            # Initiate optimizer
            optimizer = moi_optimizer()
            # Pass the problem to the ExaPF.MOIEvaluator
            moi_solution = ExaPF.optimize!(optimizer, aug, uk)
            MOI.empty!(optimizer)
            solution = (
                status=moi_solution.status,
                minimizer=moi_solution.minimizer,
                inf_du=1e8,
                iter=100,
            )
        end
        uk = solution.minimizer
        norm_grad = solution.inf_du
        n_iter = solution.iter

        # Evaluate current position in the original space
        ExaPF.constraint!(nlp, cons, uk)
        obj = ExaPF.objective(nlp, uk)
        inf_pr = ExaPF.primal_infeasibility!(nlp, cons, uk)

        verbose && log_iter(i_out, obj, inf_pr, norm_grad, ηk, n_iter) # Log evolution
        push!(tracer, obj, inf_pr, norm_grad)

        if (norm_grad < ωtol) && (inf_pr < 1e-8)
            break
        end

        # Update starting point
        u_start .= uk
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
        status=MOI.OPTIMAL,
        minimum=obj,
        minimizer=uk,
        trace=tracer,
    )

    return solution
end

