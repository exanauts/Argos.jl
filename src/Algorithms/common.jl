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
function number_active(u::VT, u♭::VT, u♯::VT; tol=1e-12) where VT<:AbstractArray
    num = 0
    for i in eachindex(u)
        if (u[i] < u♭[i] + tol) || (u[i] > u♯[i] - tol)
            num += 1
        end
    end
    return num
end

function active!(w::VT, u::VT, u♭::VT, u♯::VT; tol=1e-12) where VT<:AbstractArray
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
function project!(y, x, x♭, x♯)
    y .= max.(min.(x, x♯), x♭)
end

function project_step!(y, x, d, x♭, x♯, α)
    y .= x .+ α .* d
    y .= max.(min.(y, x♯), x♭)
end

## Feasible direction
function feasible_direction!(d, w, u, ∇f, α, u♭, u♯)
    project!(w, u .- α .* ∇f, u♭, u♯)
    # Feasible direction
    d .= w .- u
    return
end

## Printing procedure
function log_header()
    @printf(
        "iter    objective    inf_pr   inf_du    η         ρ   #inner\n"
    )
end

function log_iter(nit, obj, inf_pr, inf_du, alpha, rho, n_inner)
    @printf(
        "%4i %10.7e %6.2e %6.2e %6.2e %6.2e %i\n",
        nit, obj, inf_pr, inf_du, alpha, rho, n_inner
    )
end

