
struct RF{T} <: LinearAlgebra.Factorization
    rf::RfLU{T}
    n::Int
    # M = L + U
    M::CuSparseMatrixCSR{T, Int32}
    # Permutation matrices
    P::CuSparseMatrixCSR{T, Int32}
    Q::CuSparseMatrixCSR{T, Int32}
    # buffers
    r::CuVector{T}
    T::CuMatrix{T}
end

Base.size(rf::RF) = size(rf.M)
Base.size(rf::RF, dim::Integer) = size(rf.M, dim)

adjoint(rf::RF) = Adjoint(rf)

function RF(J::CuSparseMatrixCSR{T}; nbatch=1) where T <: Float64
    rf = CUSOLVERRF.rflu(J)
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

    r = CuVector{Float64}(undef, n)
    T = CuMatrix{Float64}(undef, n, nbatch)

    return RF(rf, n, M, P, Q, r, T)
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
    mul!(Z, P, X)
    CUSPARSE.sm2!('N', 'L', 'U', 1.0, M, Z, 'O')
    CUSPARSE.sm2!('N', 'U', 'N', 1.0, M, Z, 'O')
    mul!(Y, Q, Z)
end

# Backward solve
function LinearAlgebra.ldiv!(
    y::AbstractVector{T}, rf::Adjoint{T, RF{T}}, x::AbstractVector{T},
) where T
    z = rf.r
    mul!(z, Q', x)
    CUSPARSE.sv2!('T', 'U', 'N', 1.0, M, z, 'O')
    CUSPARSE.sv2!('T', 'L', 'U', 1.0, M, z, 'O')
    mul!(y, P', z)
end

function LinearAlgebra.ldiv!(
    Y::AbstractMatrix{T}, rf::Adjoint{T, RF{T}}, X::AbstractMatrix{T},
) where T
    @assert size(Y, 2) == size(X, 2) == size(rf.T, 2)
    Z = rf.T
    mul!(Z, Q', X)
    CUSPARSE.sm2!('T', 'U', 'N', 1.0, M, Z, 'O')
    CUSPARSE.sm2!('T', 'L', 'U', 1.0, M, Z, 'O')
    mul!(Y, P', Z)
end

