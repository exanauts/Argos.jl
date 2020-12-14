
Base.@kwdef struct AugLagSolver <: AbstractExaOptimizer
    max_iter::Int = 1_000
    ρ0::Float64 = 0.1
    ωtol::Float64 = 1e-5
    α0::Float64 = 1.0
    verbose::Int = 0
end
# TODO: add scaling option

function ExaPF.optimize!(
    algo::AugLagSolver,
    nlp::ExaPF.AbstractNLPEvaluator,
    u0::AbstractVector,
)
    aug = ExaPF.AugLagEvaluator(nlp, u0; scale=true)
    return ExaPF.optimize!(algo, aug, u0)
end

# Augmented Lagrangian method
function ExaPF.optimize!(
    algo::AugLagSolver,
    aug::ExaPF.AugLagEvaluator,
    u0::AbstractVector,
)
    # Initialize arrays
    uk        = copy(u0)
    u_start   = copy(u0)
    wk        = copy(u0)
    u_prev    = copy(u0)
    grad      = similar(u0)
    ut        = similar(u0)
    obj = Inf
    fill!(grad, 0)
    norm_grad = Inf
    nlp = aug.inner
    m = ExaPF.n_constraints(nlp)
    cons = similar(u0, m)

    tracer = Tracer()

    c0 = algo.ρ0
    ωtol = algo.ωtol
    α0 = algo.α0
    verbose = (algo.verbose > 0)

    ηk = 1.0 / (c0^0.1)

    verbose && log_header()
    for i_out in 1:algo.max_iter
        uk .= u_start
        # Inner iteration: projected gradient algorithm
        # solution = ngpa(aug, uk; α_bb=α0, α♯=α0, tol=ωtol)
        solution = tron_solve(aug, uk;
                              options=Dict("max_minor" => 2000,
                                           "max_feval" => 3000,
                                           "tol" => 1e-3))
        uk = solution.minimizer
        norm_grad = solution.inf_du
        n_iter = solution.iter

        # Evaluate current position in the original space
        ExaPF.constraint!(nlp, cons, uk)
        obj = ExaPF.objective(nlp, uk)
        inf_pr = ExaPF.primal_infeasibility!(nlp, cons, uk)

        verbose && log_iter(i_out, obj, inf_pr, norm_grad, ηk, n_iter) # Log evolution
        push!(tracer, obj, inf_pr, norm_grad)

        if (norm_grad < 1e-5) && (inf_pr < 1e-8)
            break
        end

        # Update starting point
        u_start .= uk
        # Update the penalties (see Nocedal & Wright, page 521)
        if norm(abs.(aug.infeasibility), Inf) <= ηk
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

