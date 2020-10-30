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

function active!(w::VT, u::VT, u♭::VT, u♯::VT; tol=1e-8) where VT<:AbstractArray
    @assert length(w) == length(u)
    for i in eachindex(u)
        if (u[i] < u♭[i] + tol) || (u[i] > u♯[i] - tol)
            w[i] = 0
        end
    end
end

function exaflag()
    @printf(
        "iter    objective    inf_pr   inf_du  alpha   #asa\n"
    )
end
function exaprint(nit, obj, inf_pr, inf_du, alpha, n_inner)
    @printf(
        "%4i %10.7e %6.2e %6.2e %6.2e  %i\n",
        nit, obj, inf_pr, inf_du, alpha, n_inner
    )
end


