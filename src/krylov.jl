
struct KrylovWrapper{T, VT, SMT, K}
    Gx::SMT
    solver::K
end

function KrylovWrapper{T, VT}(Gx) where {T, VT}
    nx = size(Gx, 1)
    solver = Krylov.BicgstabSolver(nx, nx, VT)
    return KrylovWrapper{T, VT, typeof(Gx), typeof(solver)}(Gx, solver)
end

Base.size(kw::KrylovWrapper) = size(kw.Gx)
Base.size(kw::KrylovWrapper, dim::Integer) = size(kw.Gx, dim)
# Lazy-adjoint
LinearAlgebra.adjoint(kw::KrylovWrapper{T}) where T = LinearAlgebra.Adjoint{T, typeof(kw)}(kw)

function LinearAlgebra.ldiv!(kw::KrylovWrapper{T}, x::AbstractVector{T}) where T
    Krylov.solve!(kw.solver, kw.Gx, x; atol=1e-10, rtol=1e-10)
    copyto!(x, kw.solver.x)
    return x
end

function LinearAlgebra.ldiv!(akw::LinearAlgebra.Adjoint{T, KW}, x::AbstractVector{T}) where {T, KW<:KrylovWrapper{T}}
    kw = parent(akw)
    Krylov.solve!(kw.solver, kw.Gx', x; atol=1e-10, rtol=1e-10)
    copyto!(x, kw.solver.x)
    return x
end

function LinearAlgebra.ldiv!(kw::KrylovWrapper{T}, X::AbstractMatrix{T}) where T
    n, m = size(X)
    for k in 1:m
        x = view(X, :, k)
        Krylov.solve!(kw.solver, kw.Gx, x; atol=1e-10, rtol=1e-10)
        copyto!(x, kw.solver.x)
    end
    return X
end

function LinearAlgebra.ldiv!(akw::LinearAlgebra.Adjoint{T, KW}, X::AbstractMatrix{T}) where {T, KW<:KrylovWrapper{T}}
    kw = parent(akw)
    n, m = size(X)
    for k in 1:m
        x = view(X, :, k)
        Krylov.solve!(kw.solver, kw.Gx', x; atol=1e-10, rtol=1e-10)
        copyto!(x, kw.solver.x)
    end
    return X
end

