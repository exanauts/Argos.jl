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
