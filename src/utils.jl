
function _copyto!(dest::AbstractArray, off1, src::AbstractArray, off2, n)
    copyto!(dest, off1, src, off2, n)
end
function _copyto!(dest::AbstractArray, off1, src::SubArray, off2, n)
    p_src = parent(src)
    copyto!(dest, off1, p_src, off2 + src.offset1, n)
end
function _copyto!(dest::SubArray, off1, src::AbstractArray, off2, n)
    p_dest = parent(dest)
    copyto!(p_dest, off1 + dest.offset1, src, off2, n)
end

# Counters
abstract type AbstractCounter end

mutable struct NLPTimers
    update_time::Float64
    obj_time::Float64
    cons_time::Float64
    grad_time::Float64
    jacobian_time::Float64
    jacprod_time::Float64
    hessian_time::Float64
    hessprod_time::Float64
end
NLPTimers() = NLPTimers(0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0)

mutable struct NLPCounter <: AbstractCounter
    objective::Int
    gradient::Int
    hessian::Int
    jacobian::Int
    jtprod::Int
    hprod::Int
end
NLPCounter() = NLPCounter(0, 0, 0, 0, 0, 0)

function Base.empty!(c::NT) where {NT <: Union{NLPTimers,NLPCounter}}
    for attr in fieldnames(NT)
        setfield!(c, attr, 0)
    end
end

function Base.sum(c::NLPTimers)
    acc = 0.0
    for attr in fieldnames(NLPTimers)
        acc += getfield(c, attr)
    end
    return acc
end

# Tracker
struct NLPTracker{VT}
    bounds::Tuple{VT, VT}
    g::VT
    d::VT
    dual_infeasibility::Vector{Float64}
    primal_infeasibility::Vector{Float64}
    cv::Tuple{VT, VT, VT}
    inf_voltage::Vector{Float64}
    cp::Tuple{VT, VT, VT}
    inf_active_power::Vector{Float64}
    cq::Tuple{VT, VT, VT}
    inf_reactive_power::Vector{Float64}
    cl::Tuple{VT, VT, VT}
    inf_line_constraints::Vector{Float64}
    pf::VT
    inf_power_flow::Vector{Float64}
    ext::Dict
end

function transfer_auglag_hessian!(
    dest::AbstractMatrix{T},
    H::AbstractMatrix{T},
    J::AbstractMatrix{T},
    ρ::AbstractVector{T},
) where T
    n = size(H, 1)
    m = size(J, 1)
    @assert size(dest, 1) == n + m
    @assert size(ρ, 1) == m

    @views begin
        Hᵤᵤ = dest[1:n, 1:n]
        Hᵤᵥ = dest[1:n, 1+n:end]
        Hᵥᵤ = dest[1+n:end, 1:n]
        Hᵥᵥ = dest[1+n:end, 1+n:end]
    end
    # Block (1, 1)
    copyto!(Hᵤᵤ, H)

    D = Diagonal(ρ)
    mul!(Hᵤᵥ, J', -D)
    mul!(Hᵥᵤ, - D, J)
    fill!(Hᵥᵥ, 0)
    ind = diagind(Hᵥᵥ) # extract coefficients on the diagonal
    Hᵥᵥ[ind] .= ρ
    return
end

