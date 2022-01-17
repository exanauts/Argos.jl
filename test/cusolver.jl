
using LinearAlgebra
using CUDAKernels
using BlockPowerFlow

using KernelAbstractions
using CUDA
using CUDA.CUSPARSE
using ExaPF
import ExaPF: LinearSolvers
import BlockPowerFlow: CUSOLVERRF

const LS = LinearSolvers

# Overload factorization routine to use cusolverRF
LS.DirectSolver(J::CuSparseMatrixCSR) = LS.DirectSolver(CUSOLVERRF.CusolverRfLU(J))

function LS.update!(s::LS.DirectSolver{Fac}, J::CuSparseMatrixCSR) where {Fac <: Factorization}
    lu!(s.factorization, J) # Update factorization inplace with transpose matrix
end

function LS.rdiv!(s::LS.DirectSolver{Fac}, y::CuVector, J::CuSparseMatrixCSR, x::CuVector) where {Fac <: CUSOLVERRF.CusolverRfLU}
    Jt = CuSparseMatrixCSC(J) # Transpose of CSR is CSC
    lu!(s.factorization, Jt) # Update factorization inplace with transpose matrix
    LinearAlgebra.ldiv!(y, s.factorization, x) # Forward-backward solve
    return 0
end

# Overload factorization for batch Hessian computation
function Argos.batch_factorization(J::CuSparseMatrixCSR, nbatch)
    Jtrans = CUSPARSE.CuSparseMatrixCSC(J)
    if nbatch == 1
        lufac = CUSOLVERRF.CusolverRfLU(J)
        lufact = CUSOLVERRF.CusolverRfLU(Jtrans)
    else
        lufac = CUSOLVERRF.CusolverRfLUBatch(J, nbatch)
        lufact = CUSOLVERRF.CusolverRfLUBatch(Jtrans, nbatch)
    end
    return (lufac, lufact)
end

function Argos.update_factorization!(hlag::Argos.AbstractReduction, J::CUSPARSE.CuSparseMatrixCSR)
    LinearAlgebra.lu!(hlag.lu, J)
    ∇gₓᵀ = CUSPARSE.CuSparseMatrixCSC(J)
    LinearAlgebra.lu!(hlag.adjlu, ∇gₓᵀ)
    return
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
@kernel function _batch_tangents_kernel2!(seeds, offset, n_batches)
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
    ev = _batch_tangents_kernel2!(CPU())(seeds, offset, n_batches, ndrange=ndrange)
    wait(ev)
    return
end

@kernel function _tgtmul_1_kernel4!(y, A_rowPtr, A_colVal, A_nzVal, z, w, nx, nu)
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


@kernel function _tgtmul_2_kernel!(yx, yu, A_rowPtr, A_colVal, A_nzVal, z, w, nx, nu)
    i, k = @index(Global, NTuple)
    @inbounds for c in A_rowPtr[i]:A_rowPtr[i+1]-1
        j = A_colVal[c]
        if (i <= nx) && (j <= nx)
            @inbounds yx[i, k] += A_nzVal[c] * z[j, k]
        elseif (i <= nx) && (j <= nx + nu)
            @inbounds yx[i, k] += A_nzVal[c] * w[j - nx, k]
        elseif (i <= nx + nu) && (j <= nx)
            @inbounds yu[i - nx, k] += A_nzVal[c] * z[j, k]
        elseif (i <= nx + nu) && (j <= nx + nu)
            @inbounds yu[i - nx, k] += A_nzVal[c] * w[j - nx, k]
        end
    end
end

function Argos.tgtmul!(y::AbstractArray, A::CuSparseMatrixCSR, z::AbstractArray, w::AbstractArray)

    n, m = size(A)
    nz, nw = size(z, 1), size(w, 1)
    @assert m == nz + nw
    @assert size(z, 2) == size(w, 2) == size(y, 2)
    k = size(z, 2)
    ndrange = (n, k)
    fill!(y, 0)
    ev = _tgtmul_1_kernel4!(CUDADevice())(
        y, A.rowPtr, A.colVal, A.nzVal, z, w, nz, nw;
        ndrange=ndrange, dependencies=Event(CUDADevice()),
    )
    wait(ev)
end

function Argos.tgtmul!(yx::AbstractArray, yu::AbstractArray, A::CuSparseMatrixCSR, z::AbstractArray, w::AbstractArray)
    n, m = size(A)
    nz, nw = size(z, 1), size(w, 1)
    @assert m == nz + nw
    @assert size(z, 2) == size(w, 2) == size(yx, 2) == size(yu, 2)
    k = size(z, 2)
    ndrange = (n, k)
    fill!(yx, 0)
    fill!(yu, 0)
    ev = _tgtmul_2_kernel!(CUDADevice())(
        yx, yu, A.rowPtr, A.colVal, A.nzVal, z, w, nz, nw;
        ndrange=ndrange, dependencies=Event(CUDADevice()),
    )
    wait(ev)
end

