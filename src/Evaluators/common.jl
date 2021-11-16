# Common interface for AbstractNLPEvaluator
#
function Base.show(io::IO, nlp::AbstractNLPEvaluator)
    n = n_variables(nlp)
    m = n_constraints(nlp)
    println(io, "A Evaluator object")
    println(io, "    * #vars: ", n)
    println(io, "    * #cons: ", m)
end

## Generic callbacks
function constraint(nlp::AbstractNLPEvaluator, x)
    cons = similar(x, n_constraints(nlp)) ; fill!(cons, 0)
    constraint!(nlp, cons, x)
    return cons
end

function gradient(nlp::AbstractNLPEvaluator, x)
    ∇f = similar(x) ; fill!(∇f, 0)
    gradient!(nlp, ∇f, x)
    return ∇f
end

function jacobian(nlp::AbstractNLPEvaluator, x)
    n = n_variables(nlp)
    m = n_constraints(nlp)
    J = similar(x, m, n) ; fill!(J, 0)
    jacobian!(nlp, J, x)
    return J
end

# Default implementation of jprod!, using full Jacobian matrix
function jprod!(nlp::AbstractNLPEvaluator, jv, u, v)
    nᵤ = length(u)
    m  = n_constraints(nlp)
    @assert nᵤ == length(v)
    jac = jacobian(nlp, u)
    mul!(jv, jac, v)
    return
end

# Joint Objective Jacobian transpose vector product (default implementation)
function ojtprod!(nlp::AbstractNLPEvaluator, jv, u, σ, v)
    gradient!(nlp, jv, u)
    jv .*= σ  # scale gradient
    jtprod!(nlp, jv, u, v)
    return
end

# Generate Hessian using Hessian-vector product routine
macro define_hessian(function_name, target_function, args...)
    fname = Symbol(function_name)
    argstup = Tuple(args)
    quote
        function $(esc(fname))(nlp::AbstractNLPEvaluator, hess, $(map(esc, argstup)...))
            @assert has_hessian(nlp)
            n = n_variables(nlp)
            v = similar(x)
            @inbounds for i in 1:n
                hv = @view hess[:, i]
                fill!(v, 0)
                v[i:i] .= 1.0
                $target_function(nlp, hv, $(map(esc, argstup)...), v)
            end
        end
    end
end

@define_hessian hessian! hessprod! x
@define_hessian hessian_lagrangian_penalty! hessian_lagrangian_penalty_prod! x y σ D

function hessian(nlp::AbstractNLPEvaluator, x)
    n = n_variables(nlp)
    H = similar(x, n, n) ; fill!(H, 0)
    hessian!(nlp, H, x)
    return H
end

function set_batch_tangents!(seeds, offset, n, n_batches)
    @assert size(seeds) == (n, n_batches)
    fill!(seeds, 0.0)
    @inbounds for j in 1:n_batches
        seeds[j+offset, j] = 1.0
    end
end

# Counters
abstract type AbstractCounter end

mutable struct NLPCounter <: AbstractCounter
    objective::Int
    gradient::Int
    hessian::Int
    jacobian::Int
    jtprod::Int
    hprod::Int
end
NLPCounter() = NLPCounter(0, 0, 0, 0, 0, 0)

function Base.empty!(c::NLPCounter)
    for attr in fieldnames(NLPCounter)
        setfield!(c, attr, 0)
    end
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

function NLPTracker(nlp::AbstractNLPEvaluator)
    model = backend(nlp)
    buffer = Base.get(nlp, ExaPF.PhysicalState())

    u_L, u_U = bounds(nlp, Variables())
    g = similar(u_L)
    d = similar(u_L)
    v_lb, v_ub = ExaPF.bounds(model, ExaPF.voltage_magnitude_constraints)
    cv = similar(v_lb)
    p_lb, p_ub = ExaPF.bounds(model, ExaPF.active_power_constraints)
    cp = similar(p_lb)
    q_lb, q_ub = ExaPF.bounds(model, ExaPF.reactive_power_constraints)
    cq = similar(q_lb)
    l_lb, l_ub = ExaPF.bounds(model, ExaPF.flow_constraints)
    cl = similar(l_lb) ; fill!(cl, 0.0)
    pf = similar(buffer.dx)

    return NLPTracker{typeof(u_L)}(
        (u_L, u_U), g, d,
        Float64[], Float64[],
        (cv, v_lb, v_ub), Float64[],
        (cp, p_lb, p_ub), Float64[],
        (cq, q_lb, q_ub), Float64[],
        (cl, l_lb, l_ub), Float64[],
        pf, Float64[],
        Dict{Symbol, Any}(:it=>1, :current_iter=>Int[], :scale_sd=>1.0),
    )
end

function store!(aug::AbstractNLPEvaluator, track::NLPTracker, u::AbstractArray)
    # Load model
    model = backend(aug)
    buffer = Base.get(aug, ExaPF.PhysicalState())
    # Load working arrays
    u_lb, u_ub = track.bounds
    cv, v_lb, v_ub = track.cv
    cp, p_lb, p_ub = track.cp
    cq, q_lb, q_ub = track.cq
    cl, l_lb, l_ub = track.cl
    # Evaluate gradient
    gradient!(aug, track.g, u)
    feasible_direction!(track.d, u, track.g, 1.0, u_lb, u_ub)
    # Evaluate constraints
    ExaPF.voltage_magnitude_constraints(model, cv, buffer)
    ExaPF.active_power_constraints(model, cp, buffer)
    ExaPF.reactive_power_constraints(model, cq, buffer)
    ExaPF.flow_constraints(model, cl, buffer)
    ExaPF.power_balance(model, track.pf, buffer)
    # Store results
    push!(track.primal_infeasibility, norm(aug.cons, Inf))
    push!(track.inf_voltage, max_infeasibility_rel(cv, v_lb, v_ub))
    push!(track.inf_active_power, max_infeasibility_rel(cp, p_lb, p_ub))
    push!(track.inf_reactive_power, max_infeasibility_rel(cq, q_lb, q_ub))
    push!(track.inf_line_constraints, max_infeasibility_rel(cl, l_lb, l_ub))
    push!(track.inf_power_flow, norm(track.pf, Inf))
    push!(track.dual_infeasibility, norm(track.d, Inf) / track.ext[:scale_sd])

    push!(track.ext[:current_iter], track.ext[:it])

    return
end

# Active set utils
function _check(val, val_min, val_max)
    violated_inf = findall(val .< val_min)
    violated_sup = findall(val .> val_max)
    n_inf = length(violated_inf)
    n_sup = length(violated_sup)
    err_inf = norm(val_min[violated_inf] .- val[violated_inf], Inf)
    err_sup = norm(val[violated_sup] .- val_max[violated_sup], Inf)
    return (n_inf, err_inf, n_sup, err_sup)
end

function _inf_pr(nlp::AbstractNLPEvaluator, cons)
    (n_inf, err_inf, n_sup, err_sup) = _check(cons, nlp.g_min, nlp.g_max)
    return max(err_inf, err_sup)
end

#=
    HESSIAN
=#
# Small utils to compute the factorization for batch Hessian algorithm
function _batch_hessian_factorization(J::AbstractSparseMatrix, nbatch)
    lufac = lu(J)
    if isnothing(lufac)
        error("Unable to find a factorization routine for type $(typeof(J))")
    end
    return (lufac, lufac')
end

abstract type AbstractHessianStorage end

struct HessianLagrangian{VT,Hess,Fac1,Fac2} <: AbstractHessianStorage
    hess::Hess
    # Adjoints
    y::VT
    z::VT
    ψ::VT
    # Buffers
    _w1::VT
    # Tangents
    tmp_tgt::VT
    tmp_hv::VT
    lu::Fac1
    adjlu::Fac2
end
function HessianLagrangian(polar::PolarForm{T, VI, VT, MT}, func::Function, J::AbstractSparseMatrix, ncons::Int) where {T, VI, VT, MT}
    lu1, lu2 = _batch_hessian_factorization(J, 1)
    nx, nu = ExaPF.get(polar, ExaPF.NumberOfState()), ExaPF.get(polar, ExaPF.NumberOfControl())
    m = ExaPF.size_constraint(polar, func)::Int
    H = AutoDiff.Hessian(polar, func)
    y = VT(undef, m)
    z = VT(undef, nx)
    ψ = VT(undef, nx)
    tgt = VT(undef, nx+nu)
    hv = VT(undef, nx+nu)
    _w1 = VT(undef, ncons)
    return HessianLagrangian(H, y, z, ψ, _w1, tgt, hv, lu1, lu2)
end
n_batches(hlag::HessianLagrangian) = 1

struct BatchHessianLagrangian{MT,Hess,Fac1,Fac2} <: AbstractHessianStorage
    nbatch::Int
    hess::Hess
    # Adjoints
    y::MT
    z::MT
    ψ::MT
    # Buffer
    _w1::MT
    _w2::MT
    _w3::MT
    # Tangents
    tangents::MT
    tmp_tgt::MT
    tmp_hv::MT
    lu::Fac1
    adjlu::Fac2
end
function BatchHessianLagrangian(polar::PolarForm{T, VI, VT, MT}, func::Function, J, nbatch, ncons) where {T, VI, VT, MT}
    lu1, lu2 = _batch_hessian_factorization(J, nbatch)
    nx, nu = ExaPF.get(polar, ExaPF.NumberOfState()), ExaPF.get(polar, ExaPF.NumberOfControl())
    m = ExaPF.size_constraint(polar, func)::Int
    H = ExaPF.BatchHessian(polar, func, nbatch)
    y   = MT(undef, m, 1)  # adjoint is the same for all batches
    z   = MT(undef, nx, nbatch)
    ψ   = MT(undef, nx, nbatch)
    tgt = MT(undef, nx+nu, nbatch)
    hv  = MT(undef, nx+nu, nbatch)
    v  = MT(undef, nu, nbatch)
    _w1 = MT(undef, ncons, nbatch)
    _w2 = MT(undef, nx, nbatch)
    _w3 = MT(undef, nu, nbatch)
    return BatchHessianLagrangian(nbatch, H, y, z, ψ, _w1, _w2, _w3, v, tgt, hv, lu1, lu2)
end
n_batches(hlag::BatchHessianLagrangian) = hlag.nbatch

function update_factorization!(hlag::AbstractHessianStorage, J::AbstractSparseMatrix)
    LinearAlgebra.lu!(hlag.lu, J)
    return
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
