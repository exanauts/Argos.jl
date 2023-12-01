struct KrylovWrapper{T, SMT, K1, K2, P}
    Gx::SMT
    solver1::K1
    solver2::K2
    preconditioner::P
end

function KrylovWrapper{T, VT, MT}(Gx, n, p) where {T, VT, MT}
    solver1 = ExaPF.BlockGmresSolver(n, n, p, 1, VT, MT)
    solver2 = Krylov.GmresSolver(n, n, 1, VT)
    device = KernelAbstractions.get_backend(solver1.X)
    preconditioner = KP.BlockJacobiPreconditioner(Gx, 32, device)
    return KrylovWrapper{T, typeof(Gx), typeof(solver1), typeof(solver2), typeof(preconditioner)}(Gx, solver1, solver2, preconditioner)
end

Base.size(kw::KrylovWrapper) = size(kw.Gx)
Base.size(kw::KrylovWrapper, dim::Integer) = size(kw.Gx, dim)
# Lazy-adjoint
LinearAlgebra.adjoint(kw::KrylovWrapper{T}) where T = LinearAlgebra.Adjoint{T, typeof(kw)}(kw)

function LinearAlgebra.ldiv!(kw::KrylovWrapper{T}, x::AbstractVector{T}) where T
    Krylov.solve!(kw.solver2, kw.Gx, x; N=kw.preconditioner, verbose=0, atol=1e-8, rtol=0.0)
    copyto!(x, kw.solver2.x)
    return x
end

function LinearAlgebra.ldiv!(akw::LinearAlgebra.Adjoint{T, KW}, x::AbstractVector{T}) where {T, KW<:KrylovWrapper{T}}
    kw = parent(akw)
    Krylov.solve!(kw.solver2, kw.Gx', x; N=kw.preconditioner, verbose=0, atol=1e-8, rtol=0.0)
    copyto!(x, kw.solver2.x)
    return x
end

function LinearAlgebra.ldiv!(kw::KrylovWrapper{T}, X::AbstractMatrix{T}) where T
    ExaPF.block_gmres!(kw.solver1, kw.Gx, X; N=kw.preconditioner, verbose=0, atol=1e-8, rtol=0.0)
    copyto!(X, kw.solver1.X)
    return X
end

function LinearAlgebra.ldiv!(akw::LinearAlgebra.Adjoint{T, KW}, X::AbstractMatrix{T}) where {T, KW<:KrylovWrapper{T}}
    kw = parent(akw)
    ExaPF.block_gmres!(kw.solver1, kw.Gx', X; N=kw.preconditioner, verbose=0, atol=1e-8, rtol=0.0)
    copyto!(X, kw.solver1.X)
    return X
end
