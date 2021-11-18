
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

struct FullHessianLagrangian{MT,VI,VT,Hess} <: AbstractHessianStorage
    # Autodiff Backend
    hess::Hess
    # Hessian in COO format
    h_I::VI
    h_J::VI
    h_V::VT
    # Coloring
    ncolors::Int
    coloring::VI
    compressedH::MT
    seeds::MT
    # cache
    y::VT
    tgt::VT
    hv::VT
end

function FullHessianLagrangian(polar::PolarForm{T, VI, VT, MT}, func::Function, buffer) where {T, VI, VT, MT}
    nx, nu = ExaPF.get(polar, ExaPF.NumberOfState()), ExaPF.get(polar, ExaPF.NumberOfControl())
    m = ExaPF.size_constraint(polar, func)::Int
    nv = nx + nu

    # Init AutoDiff
    hess_ad = AutoDiff.Hessian(polar, func)
    y = VT(undef, m)
    tgt = VT(undef, nv)
    hv = VT(undef, nv)

    nbus = ExaPF.get(polar, PS.NumberOfBuses())
    λ = ones(nbus)
    copyto!(y, randn(m))
    # Run sparsity detection on MATPOWER Hessian matrix
    H_mat = ExaPF.hessian_sparsity(polar, func, buffer, λ)
    H = [H_mat.xx H_mat.xu' ; H_mat.xu H_mat.uu]::SparseMatrixCSC{ComplexF64, Int}
    # Add correction associated to slack's power generation
    Jx = ExaPF.jacobian_sparsity(polar, ExaPF.active_power_constraints, State())
    Ju = ExaPF.jacobian_sparsity(polar, ExaPF.active_power_constraints, Control())
    J = [Jx Ju]
    H += J' * J

    # Coloring
    colors = ExaPF.AutoDiff.SparseDiffTools.matrix_colors(H)
    ncolors = length(unique(colors))
    compressedH = MT(undef, nv, ncolors)
    fill!(compressedH, 0.0)

    # Seeds
    seeds = MT(undef, nv, ncolors)
    fill!(seeds, 0.0)
    for i in 1:nv
        @inbounds for j in 1:ncolors
            if colors[i] == j
                seeds[i, j] = 1.0
            end
        end
    end

    # Compute sparsity pattern on lower triangular part
    LH = SparseMatrixCSC(LowerTriangular(H))
    I, J, V = SparseArrays.findnz(LH)
    V = real.(V)

    return FullHessianLagrangian{MT, typeof(I), typeof(V), typeof(hess_ad)}(
        hess_ad, I, J, V, ncolors, colors, compressedH, seeds, y, tgt, hv,
    )
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
