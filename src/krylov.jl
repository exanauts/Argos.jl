struct KrylovWrapper{T, SMT, K1, K2}
    Gx::SMT
    solver1::K1
    solver2::K2
end

function KrylovWrapper{T, VT, MT}(Gx, n, p) where {T, VT, MT}
    solver1 = ExaPF.BlockGmresSolver(n, n, p, 1, VT, MT)
    solver2 = Krylov.GmresSolver(n, n, 1, VT)
    return KrylovWrapper{T, typeof(Gx), typeof(solver1), typeof(solver2)}(Gx, solver1, solver2)
end

Base.size(kw::KrylovWrapper) = size(kw.Gx)
Base.size(kw::KrylovWrapper, dim::Integer) = size(kw.Gx, dim)
# Lazy-adjoint
LinearAlgebra.adjoint(kw::KrylovWrapper{T}) where T = LinearAlgebra.Adjoint{T, typeof(kw)}(kw)

function LinearAlgebra.ldiv!(kw::KrylovWrapper{T}, x::AbstractVector{T}) where T
    Krylov.solve!(kw.solver2, kw.Gx, x; verbose=1, atol=1e-8, rtol=0.0)
    copyto!(x, kw.solver2.x)
    return x
end

function LinearAlgebra.ldiv!(akw::LinearAlgebra.Adjoint{T, KW}, x::AbstractVector{T}) where {T, KW<:KrylovWrapper{T}}
    kw = parent(akw)
    Krylov.solve!(kw.solver2, kw.Gx', x; verbose=1, atol=1e-8, rtol=0.0)
    copyto!(x, kw.solver2.x)
    return x
end

function LinearAlgebra.ldiv!(kw::KrylovWrapper{T}, X::AbstractMatrix{T}) where T
    ExaPF.block_gmres!(kw.solver1, kw.Gx, X; verbose=1, atol=1e-10, rtol=1e-10)
    copyto!(X, kw.solver1.X)
    return X
end

function LinearAlgebra.ldiv!(akw::LinearAlgebra.Adjoint{T, KW}, X::AbstractMatrix{T}) where {T, KW<:KrylovWrapper{T}}
    kw = parent(akw)
    ExaPF.block_gmres!(kw.solver1, kw.Gx', X; verbose=1, atol=1e-10, rtol=1e-10)
    copyto!(X, kw.solver1.X)
    return X
end
