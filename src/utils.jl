import Base: push!

#
struct Tracer{T}
    objective::AbstractVector{T}
    inf_pr::AbstractVector{T}
    inf_du::AbstractVector{T}
end

Tracer() = Tracer{Float64}([], [], [])

function push!(tracer, a, b, c)
    push!(tracer.objective, a)
    push!(tracer.inf_pr, b)
    push!(tracer.inf_du, c)
end

## Active set procedure
function active!(w::VT, u::VT, u♭::VT, u♯::VT; tol=1e-8) where VT<:AbstractArray
    @assert length(w) == length(u)
    for i in eachindex(u)
        if (u[i] < u♭[i] + tol) || (u[i] > u♯[i] - tol)
            w[i] = 0
        end
    end
end

function active_set!(A::AbstractArray, u::VT, u♭::VT, u♯::VT; tol=1e-15) where VT<:AbstractArray
    empty!(A)
    for i in eachindex(u)
        if (u[i] < u♭[i] + tol) || (u[i] > u♯[i] - tol)
            push!(A, i)
        end
    end
end

function undecided_set!(
    U::AbstractArray,
    ∇fu::VT,
    d1u::VT,
    u::VT, u♭::VT, u♯::VT;
    tol=1e-15,
    α=0.5,
    β=1.5
) where VT<:AbstractArray
    empty!(U)
    norm_d1 = norm(d1u, Inf)
    norm_d1_α = (norm_d1)^(α)
    norm_d1_β = (norm_d1)^(β)
    for i in eachindex(u)
        if (abs(∇fu[i]) >= norm_d1_α) && (u[i] - u♭[i] >= norm_d1_β) && (u♯[i] - u[i] >= norm_d1_β)
            push!(U, i)
        end
    end
end

## Line-search procedure
struct LineModel
    model::ExaPF.AbstractNLPEvaluator
    u::AbstractVector
    d::AbstractVector
    g::AbstractVector
    ut::AbstractVector
end
function (ϕ::LineModel)(α)
    ϕ.ut .= ϕ.u .+ α * ϕ.d
    ExaPF.update!(ϕ.model, ϕ.ut)
    return ExaPF.objective(ϕ.model, ϕ.ut)
end
function grad!(ϕ::LineModel, α)
    ϕ.ut .= ϕ.u .+ α * ϕ.d
    ExaPF.gradient!(ϕ.model, ϕ.g, ϕ.ut)
    return dot(ϕ.g, ϕ.d)
end

## Printing procedure
function exaflag()
    @printf(
        "iter    objective    inf_pr   inf_du  alpha     #asa\n"
    )
end
function exaprint(nit, obj, inf_pr, inf_du, alpha, n_inner)
    @printf(
        "%4i %10.7e %6.2e %6.2e %6.2e  %i\n",
        nit, obj, inf_pr, inf_du, alpha, n_inner
    )
end


