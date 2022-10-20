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

## Projection operator
function project!(y, x, x♭, x♯)
    y .= max.(min.(x, x♯), x♭)
end

# P[x - α ∇f] - x
function feasible_direction!(d, x, ∇f, α, x♭, x♯)
    project!(d, x .- α .* ∇f, x♭, x♯)  # P[x - α ∇f]
    d .-= x                            # P[x - α ∇f] - x
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

