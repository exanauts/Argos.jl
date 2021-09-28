
struct QuadraticModel{VT, MT} <: AbstractNLPEvaluator
    inner::AbstractNLPEvaluator
    H::MT
    J::MT
    gₖ::VT
    xₖ::VT
    dₖ::VT
end

function QuadraticModel(nlp::AbstractNLPEvaluator, u=initial(nlp))
    n = n_variables(nlp)
    m = n_constraints(nlp)

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
reset!(ev::QuadraticModel) = nothing

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

function refresh!(ev::QuadraticModel, x)
    copyto!(ev.xₖ, x)
    update!(ev.inner, x)
    gradient!(ev.inner, ev.gₖ, x)
    jacobian!(ev.inner, ev.J, x)
    hessian!(ev.inner, ev.H, x)
    return
end

#=
    Auglag QP model (for Schur complement method only)
=#
struct AuglagQuadraticModel{T, VT, MT} <: AbstractNLPEvaluator
    inner::AugLagEvaluator
    H::MT
    J::MT
    gₖ::VT
    xₖ::VT
    dₖ::VT
    _wx::VT
    ρ::Float64
    scaler::MaxScaler{T, VT}
end

function AuglagQuadraticModel(aug::AugLagEvaluator, x=initial(aug))
    inner = inner_evaluator(aug)
    n = n_variables(inner)
    m = n_constraints(inner)

    u = @view x[1:n]

    J = similar(x, m, n)
    H = similar(x, n, n)
    g = similar(x, n+m)
    d = similar(x, n+m)
    _wx = similar(x, n+m)

    update!(aug, x)
    gradient!(aug, g, x)

    # Jacobian
    inner_jacobian!(aug, J, u)
    # Hessian
    mask = abs.(aug.cons) .> 0
    ηcons = aug.scaler.scale_cons
    λ = aug.λc .* ηcons .* mask
    D = similar(λ) ; fill!(D, 0.0)
    inner_hessian!(aug, H, u, λ, D)

    return AuglagQuadraticModel(aug, H, J, g, x, d, _wx, aug.ρ, aug.scaler)
end

initial(ev::AuglagQuadraticModel) = ev.xₖ
n_variables(ev::AuglagQuadraticModel) = n_variables(ev.inner)
n_constraints(ev::AuglagQuadraticModel) = n_constraints(ev.inner)
inner_evaluator(ev::AuglagQuadraticModel) = inner_evaluator(ev.inner)

bounds(ev::AuglagQuadraticModel, ::Variables) = bounds(ev.inner, Variables())
bounds(ev::AuglagQuadraticModel, ::Constraints) = bounds(ev.inner, Constraints())

function Base.getproperty(ev::AuglagQuadraticModel, d::Symbol)
    if d === :inner
        return Base.getfield(ev, :inner)
    elseif (d === :λc) || (d === :tracker) || (d == :cons)
        return Base.getproperty(ev.inner, d)
    elseif hasproperty(ev, d)
        return Base.getfield(ev, d)
    else
        error("$d")
    end
end

function kkt_mul!(y::AbstractVector, ev::AuglagQuadraticModel, x::AbstractVector)
    m, n = size(ev.J)
    # Load problem
    ηcons = ev.scaler.scale_cons
    mask = abs.(ev.inner.cons) .> 0
    ρₛ = ev.ρ .* ηcons.^2 .* mask

    ds = similar(ηcons)
    x_u = @view x[1:n]
    x_s = @view x[1+n:n+m]

    y_u = @view y[1:n]
    y_s = @view y[1+n:n+m]

    # Structure of the Hessian
    #= [
       H +  ρ * J' * D² * J        - ρ J' * D²  ;
           - ρ D² J                    ρ D²
       ]
    =#

    # Block (1, 1)
    mul!(y_u, ev.H, x_u)
    mul!(ds, ev.J, x_u)
    ds .*= ρₛ
    mul!(y_u, ev.J', ds, 1.0, 1.0)

    # Block (1, 2)
    ds .= ρₛ .* x_s
    mul!(y_u, ev.J', ds, -1.0, 1.0)

    # Block (2, 1)
    mul!(ds, ev.J, x_u)

    # Block (2, 2)
    y_s .= ρₛ .* (x_s .- ds)

    return
end

function update!(ev::AuglagQuadraticModel, x)
    ev.dₖ .= x .- ev.xₖ
    return
end

function objective(ev::AuglagQuadraticModel, x)
    Hd = ev._wx
    kkt_mul!(Hd, ev, ev.dₖ)
    return 0.5 * dot(ev.dₖ, Hd) + dot(ev.gₖ, ev.dₖ)
end

function constraint!(ag::AuglagQuadraticModel, cons, u)
    @assert length(cons) == 0
    return
end

function gradient!(ev::AuglagQuadraticModel, f, x)
    kkt_mul!(f, ev, ev.dₖ)
    f .+= ev.gₖ
    return
end

jacobian!(ev::AuglagQuadraticModel, jac, x) = nothing
hessian!(ev::AuglagQuadraticModel, hess, x) = nothing

function inner_hessian!(ev::AuglagQuadraticModel, H, x, λ, D)
    copyto!(H, ev.H)
end

function inner_jacobian!(ev::AuglagQuadraticModel, J, x)
    copyto!(J, ev.J)
end

function refresh!(ev::AuglagQuadraticModel, x)
    copyto!(ev.xₖ, x)
    update!(ev.inner, x)
    gradient!(ev.inner, ev.gₖ, x)
    inner_jacobian!(ev.inner, ev.J, x)
    inner_hessian!(ev.inner, ev.H, x)
    return
end

reset!(ev::AuglagQuadraticModel) = nothing
