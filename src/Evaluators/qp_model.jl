
struct QuadraticModel{VT, MT} <: AbstractNLPEvaluator
    inner::AbstractNLPEvaluator
    H::MT
    J::MT
    gₖ::VT
    xₖ::VT
    dₖ::VT
end

function QuadraticModel(nlp::AbstractNLPEvaluator)
    n = n_variables(nlp)
    m = n_constraints(nlp)

    u = initial(nlp)

    J = similar(u, m, n)
    H = similar(u, n, n)
    g = similar(u, n)
    d = similar(u, n)

    update!(nlp, u)
    gradient!(nlp, g, u)
    jacobian!(nlp, J, u)
    hessian!(nlp, H, u)

    return QuadraticModel(nlp, H, J, g, u, d)
end

initial(ev::QuadraticModel) = ev.xₖ
n_variables(ev::QuadraticModel) = n_variables(ev.inner)
n_constraints(ev::QuadraticModel) = n_constraints(ev.inner)

bounds(ev::QuadraticModel, ::Variables) = bounds(ev.inner, Variables())
bounds(ev::QuadraticModel, ::Constraints) = bounds(ev.inner, Constraints())

function update!(ev::QuadraticModel, x)
    ev.dₖ .= x .- ev.xₖ
    return
end

function objective(ev::QuadraticModel, x)
    return 0.5 * dot(ev.dₖ, ev.H, ev.dₖ) + dot(ev.gₖ, ev.dₖ)
end

function constraint!(ag::QuadraticModel, cons, u)
    @assert length(cons) == 0
    return
end

function gradient!(ev::QuadraticModel, f, x)
    mul!(f, ev.H, ev.dₖ)
    f .+= ev.gₖ
    return
end

function jacobian!(ev::QuadraticModel, jac, x)
    copyto!(jac, ev.J)
end

function hessian!(ev::QuadraticModel, hess, x)
    copyto!(hess, ev.H)
end

# TODO: a bit hacky, but allow to embed QuadraticModel inside an AugLagEvaluator
function hessian_lagrangian_penalty!(ev::QuadraticModel, hess, x, l, σ, ρ)
    copyto!(hess, ev.H)
end

function refresh!(ev::QuadraticModel, x)
    copyto!(ev.xₖ, x)
    update!(ev.inner, x)
    gradient!(ev.inner, ev.gₖ, x)
    jacobian!(ev.inner, ev.J, x)
    hessian!(ev.inner, ev.H, x)
    return
end

