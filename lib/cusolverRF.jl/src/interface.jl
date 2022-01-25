import CUDA.CUSPARSE: CuSparseMatrixCSR
import LinearAlgebra: mul!


struct RF{Tv} <: LinearAlgebra.Factorization{Tv}
    rf::RfLU
    n::Int
    # M = L + U
    M::CuSparseMatrixCSR{Tv, Int32}
    # Permutation matrices
    P::CuSparseMatrixCSR{Tv, Int32}
    Q::CuSparseMatrixCSR{Tv, Int32}
    # buffers
    r::CuVector{Tv}
    T::CuMatrix{Tv}
    tsv::CuSparseBackSV
    dsm::CuSparseBackSM
    tsm::CuSparseBackSM
end

Base.size(rf::RF) = size(rf.M)
Base.size(rf::RF, dim::Integer) = size(rf.M, dim)

LinearAlgebra.adjoint(rf::RF) = LinearAlgebra.Adjoint(rf)

function RF(J::CuSparseMatrixCSR{Tv}; nbatch=1) where Tv <: Float64
    rf = rflu(J)
    n = size(J, 1)

    # Bundled factors M = L + U
    M = rf_extract_factors(rf, n)
    # Permutation matrices
    p = Array(rf.dP) ; p .+= Cint(1)
    q = Array(rf.dQ) ; q .+= Cint(1)
    P_cpu = sparse(1:n, p, ones(n), n, n)
    Q_cpu = sparse(q, 1:n, ones(n), n, n)
    P = CuSparseMatrixCSR(P_cpu)
    Q = CuSparseMatrixCSR(Q_cpu)

    r = CuVector{Tv}(undef, n)
    T = CuMatrix{Tv}(undef, n, nbatch)
    fill!(T, 0)

    tsv = CuSparseBackSV(M, 'T')
    dsm = CuSparseBackSM(M, 'N', T)
    tsm = CuSparseBackSM(M, 'T', T)

    return RF(rf, n, M, P, Q, r, T, tsv, dsm, tsm)
end

# Refactoring
function LinearAlgebra.lu!(rf::RF, J::CuSparseMatrixCSR)
    rf_refactor!(rf.rf, J)
end

# Direct solve
function LinearAlgebra.ldiv!(
    y::AbstractVector, rf::RF, x::AbstractVector,
)
    copyto!(y, x)
    rf_solve!(rf.rf, y)
end

function LinearAlgebra.ldiv!(
    Y::AbstractMatrix, rf::RF, X::AbstractMatrix,
)
    @assert size(Y, 2) == size(X, 2) == size(rf.T, 2)
    Z = rf.T
    mul!(Z, rf.P, X)
    backsolve!(rf.dsm, rf.M, Z)
    mul!(Y, rf.Q, Z)
end

function LinearAlgebra.ldiv!(
    rf::RF, X::AbstractMatrix,
)
    @assert size(X, 2) == size(rf.T, 2)
    Z = rf.T
    mul!(Z, rf.P, X)
    backsolve!(rf.dsm, rf.M, Z)
    mul!(X, rf.Q, Z)
end

# Backward solve
function LinearAlgebra.ldiv!(
    y::AbstractVector{T}, arf::LinearAlgebra.Adjoint{T, RF{T}}, x::AbstractVector{T},
) where T
    rf = arf.parent
    z = rf.r
    mul!(z, rf.Q', x)
    backsolve!(rf.tsv, rf.M, z)
    mul!(y, rf.P', z)
end

function LinearAlgebra.ldiv!(
    Y::AbstractMatrix{T}, arf::LinearAlgebra.Adjoint{T, RF{T}}, X::AbstractMatrix{T},
) where T
    rf = arf.parent
    @assert size(Y, 2) == size(X, 2) == size(rf.T, 2)
    Z = rf.T
    mul!(Z, rf.Q', X)
    backsolve!(rf.tsm, rf.M, Z)
    mul!(Y, rf.P', Z)
end

function LinearAlgebra.ldiv!(
    arf::LinearAlgebra.Adjoint{T, RF{T}}, X::AbstractMatrix{T},
) where T
    rf = arf.parent
    @assert size(X, 2) == size(rf.T, 2)
    Z = rf.T
    mul!(Z, rf.Q', X)
    backsolve!(rf.tsm, rf.M, Z)
    mul!(X, rf.P', Z)
end

