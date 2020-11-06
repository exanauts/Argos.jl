import Base: push!

# Convergence status
@enum OptimizationStatus begin
    NotSolved
    Unbounded
    MaxIterations
    Optimal
    Stalled
    SwitchNGPA
    SwitchCG
    Restart
end

# Tracer
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

## Projection operator
function project!(xp, x, x♭, x♯)
    xp .= max.(min.(x, x♯), x♭)
end
## Feasible direction
function feasible_direction!(d, w, u, ∇f, α, u♭, u♯)
    project!(w, u .- α .* ∇f, u♭, u♯)
    # Feasible direction
    d .= w .- u
    return
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
    ϕ.ut .= ϕ.u .+ α .* ϕ.d
    try
        ExaPF.update!(ϕ.model, ϕ.ut)
    catch
        println(ϕ.ut)
        rethrow()
    end
    return ExaPF.objective(ϕ.model, ϕ.ut)
end
function grad!(ϕ::LineModel, α)
    ϕ.ut .= ϕ.u .+ α .* ϕ.d
    ExaPF.gradient!(ϕ.model, ϕ.g, ϕ.ut)
    return dot(ϕ.g, ϕ.d)
end

function max_step(ϕ::LineModel)
    u♭, u♯ = ExaPF.bounds(ϕ.model, ExaPF.Variables())
    d, u = ϕ.d, ϕ.u
    step♭, step♯ = Inf, 0
    bdist = Inf

    for i in eachindex(ϕ.u)
        if d[i] > 0.0
            t = u♯[i] - u[i]
            bdist = min(t, bdist)
            step = t / d[i]
            step♭, step♯ = min(step♭, step), max(step♯, step)
            bdist = min(u[i] - u♭[i], bdist)
        elseif d[i] < 0.0
            t = u[i] - u♭[i]
            bdist = min(t, bdist)
            step = -t / d[i]
            step♭, step♯ = min(step♭, step), max(step♯, step)
            bdist = min(u♯[i] - u[i], bdist)
        else
            bdist = min(u[i] - u♭[i] , u♯[i] - u[i], bdist)
        end
    end

    return (bdist , step♭, step♯)
end

## Printing procedure
function log_header()
    @printf(
        "iter    objective    inf_pr   inf_du  alpha     #asa\n"
    )
end
function log_iter(nit, obj, inf_pr, inf_du, alpha, n_inner)
    @printf(
        "%4i %10.7e %6.2e %6.2e %6.2e  %i\n",
        nit, obj, inf_pr, inf_du, alpha, n_inner
    )
end

