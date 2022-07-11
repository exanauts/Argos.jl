
using Revise
using Argos

function bench_evaluator(
    nlp::Argos.AbstractNLPEvaluator;
    ntrials=5,
)
    n = Argos.n_variables(nlp)
    m = Argos.n_constraints(nlp)
    u = Argos.initial(nlp)

    Argos.update!(nlp, u)

    @info "Objective"
    Argos.objective(nlp, u)
    t_objective = 0.0
    for i in 1:ntrials
        t = @timed Argos.objective(nlp, u)
        t_objective += t.time / ntrials
    end

    @info "Gradient"
    g = similar(u)
    Argos.gradient!(nlp, g, u)
    t_gradient = 0.0
    for i in 1:ntrials
        t = @timed Argos.gradient!(nlp, g, u)
        t_gradient += t.time / ntrials
    end

    @info "Constraints"
    c = similar(u, m)
    Argos.constraint!(nlp, c, u)
    t_constraint = 0.0
    for i in 1:ntrials
        t = @timed Argos.constraint!(nlp, c, u)
        t_constraint += t.time / ntrials
    end

    @info "Jacobian"
    j_I, j_J = Argos.jacobian_structure(nlp)
    nnzJ = length(j_I)
    jvals = similar(u, nnzJ)
    Argos.jacobian_coo!(nlp, jvals, u)
    t_jacobian = 0.0
    for i in 1:ntrials
        t = @timed Argos.jacobian_coo!(nlp, jvals, u)
        t_jacobian += t.time / ntrials
    end

    @info "Hessian of Lagrangian"
    y = similar(u, m)
    σ = rand()
    copyto!(y, rand(m))
    h_I, h_J = Argos.hessian_structure(nlp)
    nnzH = length(h_I)
    hvals = similar(u, nnzH)
    Argos.hessian_lagrangian_coo!(nlp, hvals, u, y, σ)
    t_hessian = 0.0
    for i in 1:ntrials
        t = @timed Argos.hessian_lagrangian_coo!(nlp, hvals, u, y, σ)
        t_hessian += t.time / ntrials
    end

    return (
        time_objective=t_objective,
        time_gradient=t_gradient,
        time_constraint=t_constraint,
        time_jacobian=t_jacobian,
        time_hessian=t_hessian,
    )
end

