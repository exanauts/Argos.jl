
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

#=
    ExaPF.LinearSolvers
=#
ExaPF.default_sparse_matrix(::CUDADevice) = CUSPARSE.CuSparseMatrixCSR

# Overload factorization routine to use cusolverRF
LS.DirectSolver(J::CuSparseMatrixCSR) = LS.DirectSolver(CUSOLVERRF.CusolverRfLU(J))

function LS.rdiv!(s::LS.DirectSolver{Fac}, y::CuVector, J::CuSparseMatrixCSR, x::CuVector) where {Fac}
    Jt = CuSparseMatrixCSC(J) # Transpose of CSR is CSC
    lu!(s.factorization, Jt) # Update factorization inplace with transpose matrix
    LinearAlgebra.ldiv!(y, s.factorization, x) # Forward-backward solve
    return 0
end

# Overload factorization for batch Hessian computation
function ExaOpt._batch_hessian_factorization(J::CuSparseMatrixCSR, nbatch)
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

function ExaOpt.update_factorization!(hlag::ExaOpt.AbstractHessianStorage, J::CUSPARSE.CuSparseMatrixCSR)
    LinearAlgebra.lu!(hlag.lu, J)
    ∇gₓᵀ = CUSPARSE.CuSparseMatrixCSC(J)
    LinearAlgebra.lu!(hlag.adjlu, ∇gₓᵀ)
    return
end

#=
    ExaOpt.transfer_auglag_hessian
=#
@kernel function _transfer_auglag_hessian!(dest, H, J, ρ, n, m)
    i, j = @index(Global, NTuple)

    if i <= n
        dest[i, j] = H[i, j]
    elseif i <= n + m
        dest[i, j] = - ρ[i - n] * J[i - n, j]
        dest[j, i] = - ρ[i - n] * J[i - n, j]
        dest[i, i] = ρ[i - n]
    end
end

function ExaOpt.transfer_auglag_hessian!(
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
    ev = _transfer_auglag_hessian!(CUDADevice())(dest, H, J, ρ, n, m, ndrange=ndrange)
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
    ExaOpt.set_batch_tangents!
=#
@kernel function _batch_tangents_kernel!(seeds, offset, n, n_batches)
    i, j = @index(Global, NTuple)
    val = (i == j + offset) ? 1.0 : 0.0
    seeds[i, j] = val
    if (i == j + offset)
        seeds[i, j] = 1.0
    end
end

function ExaOpt.set_batch_tangents!(seeds::CuMatrix, offset, n, n_batches)
    ndrange = (n, n_batches)
    ev = _batch_tangents_kernel!(CUDADevice())(seeds, offset, n, n_batches, ndrange=ndrange)
    wait(ev)
    return
end

function test_batch_tangents!(seeds::Matrix, offset, n, n_batches)
    ndrange = (n, n_batches)
    ev = _batch_tangents_kernel!(CPU())(seeds, offset, n, n_batches, ndrange=ndrange)
    wait(ev)
    return
end
