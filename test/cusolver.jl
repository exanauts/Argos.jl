
using LinearAlgebra
using CUDAKernels

using KernelAbstractions
using CUDA
using CUDA.CUSPARSE
using ExaPF
import ExaPF: LinearSolvers

const LS = LinearSolvers

include("spgemm.jl")

# Plug cusolverRF in ExaPF.LinearSolvers
LS.DirectSolver(J::CuSparseMatrixCSR; kwargs...) = LS.DirectSolver(cusolverRF.RF(J; kwargs...))

function LS.update!(s::LS.DirectSolver{Fac}, J::CuSparseMatrixCSR) where {Fac <: Factorization}
    lu!(s.factorization, J)
end

#=
    Argos.transfer_auglag_hessian
=#
@kernel function _transfer_auglag_hessian!(dest, H, J, ρ, n, m)
    i, j = @index(Global, NTuple)

    if i <= n
        @inbounds dest[i, j] = H[i, j]
    elseif i <= n + m
        @inbounds dest[i, j] = - ρ[i - n] * J[i - n, j]
        @inbounds dest[j, i] = - ρ[i - n] * J[i - n, j]
        @inbounds dest[i, i] =   ρ[i - n]
    end
end

function Argos.transfer_auglag_hessian!(
    dest::CuMatrix{T},
    H::CuMatrix{T},
    J::CuMatrix{T},
    ρ::CuVector{T},
) where T
    n = size(H, 1)
    m = size(J, 1)
    @assert size(dest, 1) == n + m
    @assert size(ρ, 1) == m

    ndrange = (n+m, n)
    ev = _transfer_auglag_hessian!(CUDADevice())(dest, H, J, ρ, n, m, ndrange=ndrange, dependencies=Event(CUDADevice()))
    wait(ev)
    return
end

function test_transfer_auglag_hessian!(
    dest::Matrix{T},
    H::Matrix{T},
    J::Matrix{T},
    ρ::Vector{T},
) where T
    n = size(H, 1)
    m = size(J, 1)
    @assert size(dest, 1) == n + m
    @assert size(ρ, 1) == m

    ndrange = (n+m, n)
    ev = _transfer_auglag_hessian!(CPU())(dest, H, J, ρ, n, m, ndrange=ndrange)
    wait(ev)
    return
end

#=
    Argos.set_batch_tangents!
=#
@kernel function _batch_tangents_kernel!(seeds, offset, n_batches)
    i = @index(Global, Linear)
    @inbounds seeds[i + offset, i] = 1.0
end

function Argos.set_batch_tangents!(seeds::CuMatrix, offset, n, n_batches)
    @assert offset + n_batches <= n
    ndrange = (n_batches)
    fill!(seeds, 0.0)
    ev = _batch_tangents_kernel!(CUDADevice())(
        seeds, offset, n_batches;
        ndrange=ndrange, dependencies=Event(CUDADevice()),
    )
    wait(ev)
    return
end

function test_batch_tangents!(seeds::Matrix, offset, n, n_batches)
    @assert offset + n_batches <= n
    ndrange = (n_batches)
    fill!(seeds, 0.0)
    ev = _batch_tangents_kernel!(CPU())(seeds, offset, n_batches, ndrange=ndrange)
    wait(ev)
    return
end

@kernel function _tgtmul_1_kernel!(y, A_rowPtr, A_colVal, A_nzVal, z, w, nx, nu)
    i, k = @index(Global, NTuple)
    @inbounds for c in A_rowPtr[i]:A_rowPtr[i+1]-1
        j = A_colVal[c]
        if j <= nx
            @inbounds y[i, k] += A_nzVal[c] * z[j, k]
        else
            @inbounds y[i, k] += A_nzVal[c] * w[j - nx, k]
        end
    end
end


@kernel function _tgtmul_2_kernel!(yx, yu, A_rowPtr, A_colVal, A_nzVal, z, w, alpha, nx, nu)
    i, k = @index(Global, NTuple)
    @inbounds for c in A_rowPtr[i]:A_rowPtr[i+1]-1
        j = A_colVal[c]
        if (i <= nx) && (j <= nx)
            @inbounds yx[i, k] += alpha * A_nzVal[c] * z[j, k]
        elseif (i <= nx) && (j <= nx + nu)
            @inbounds yx[i, k] += alpha * A_nzVal[c] * w[j - nx, k]
        elseif (i <= nx + nu) && (j <= nx)
            @inbounds yu[i - nx, k] += alpha * A_nzVal[c] * z[j, k]
        elseif (i <= nx + nu) && (j <= nx + nu)
            @inbounds yu[i - nx, k] += alpha * A_nzVal[c] * w[j - nx, k]
        end
    end
end

function Argos.tgtmul!(
    y::AbstractArray, A::CuSparseMatrixCSR, z::AbstractArray, w::AbstractArray,
    alpha::Number, beta::Number,
)
    n, m = size(A)
    nz, nw = size(z, 1), size(w, 1)
    @assert m == nz + nw
    @assert size(z, 2) == size(w, 2) == size(y, 2)
    k = size(z, 2)
    ndrange = (n, k)
    y .*= beta
    ev = _tgtmul_1_kernel!(CUDADevice())(
        y, A.rowPtr, A.colVal, A.nzVal, z, w, alpha, nz, nw;
        ndrange=ndrange, dependencies=Event(CUDADevice()),
    )
    wait(ev)
end

function Argos.tgtmul!(
    yx::AbstractArray, yu::AbstractArray, A::CuSparseMatrixCSR, z::AbstractArray, w::AbstractArray,
    alpha::Number, beta::Number,
)
    n, m = size(A)
    nz, nw = size(z, 1), size(w, 1)
    @assert m == nz + nw
    @assert size(z, 2) == size(w, 2) == size(yx, 2) == size(yu, 2)
    k = size(z, 2)
    ndrange = (n, k)
    yx .*= beta
    yu .*= beta
    ev = _tgtmul_2_kernel!(CUDADevice())(
        yx, yu, A.rowPtr, A.colVal, A.nzVal, z, w, alpha, nz, nw;
        ndrange=ndrange, dependencies=Event(CUDADevice()),
    )
    wait(ev)
end


@kernel function _copy_index_kernel!(dest, src, idx)
    i = @index(Global, Linear)
    @inbounds dest[i] = src[idx[i]]
end

function Argos.copy_index!(dest::CuVector{T}, src::CuVector{T}, idx) where T
    @assert length(dest) == length(idx)
    ndrange = (length(dest),)
    idx_d = CuVector(idx)
    ev = _copy_index_kernel!(CUDADevice())(dest, src, idx_d; ndrange=ndrange)
    wait(ev)
end

#=
    CondensedKKT
=#
function Argos.HJDJ(W::CuSparseMatrixCSR, J::CuSparseMatrixCSR)
    n = size(W, 1)

    # Perform initial computation on the CPU
    J_h = SparseMatrixCSC(J)
    Jt_h = SparseMatrixCSC(Jh')
    JtJ_h = Jt_h * Jh

    # Compute tranpose and associated permutation
    Jt = CuSparseMatrixCSR(Jt_h)
    i, j, _ = findnz(J)
    transperm = sortperm(i) |> CuVector{Int}

    # JDJ
    JtJ = CuSparseMatrixCSR(JtJ_h)

    constants = similar(nonzeros(W), n) ; fill!(constants, 0)
    return HJDJ(W, Jt, JtJ, constants, transperm)
end


@kernel function _scale_transpose_kernel!(
    Jtz, Jp, Jj, Jz, D, tperm,
)
    i = @index(Global, Linear)

    for c in Jp[i]:Jp[i+1]-1
        j = Jj[c]
        Jtz[tperm[c]] = D[i] * Jz[c]
    end
end

function update!(K::Argos.HJDJ, A, D, Σ)
    m = size(A, 1)
    ev = _scale_transpose_kernel!(CUDADevice())(
        K.Jt.nzVal, K.J.rowPtr, K.J.colVal, K.J.nzVal, D, K.transperm,
        ndrange=(m, 1),
    )
    wait(ev)
    spgemm!('N', 'N', 1.0, K.Jt, K.J, 1.0, K.JtJ, 'O')
    K.Σ .= Σ
end

