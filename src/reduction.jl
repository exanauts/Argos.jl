
#=
    Operator S = - Gₓ⁻¹ Gᵤ
=#
abstract type SensitivityOperator end

Base.size(S::SensitivityOperator) = (S.nx, S.nu)
Base.size(S::SensitivityOperator, d) = Base.size(S)[d]

update_op!(S::SensitivityOperator) = nothing

struct ImplicitSensitivity{Fac,SMT} <: SensitivityOperator
    nx::Int
    nu::Int
    lujac::Fac
    Gu::SMT
end
function ImplicitSensitivity(lujac, Gu)
    nx, nu = size(Gu)
    return ImplicitSensitivity(nx, nu, lujac, Gu)
end

function LinearAlgebra.mul!(y::AbstractArray, S::ImplicitSensitivity, x::AbstractArray, alpha, beta)
    mul!(y, S.Gu, x, -alpha, beta)
    ldiv!(S.lujac, y)
end

# TODO: we should not modify x here
function tmul!(y::AbstractArray, S::ImplicitSensitivity, x::AbstractArray, alpha, beta) where {N, T, Imp<:ImplicitSensitivity}
    ldiv!(S.lujac', x)
    mul!(y, S.Gu', x, -alpha, beta)
end

struct DirectSensitivity{Fac,SMT,MT} <: SensitivityOperator
    nx::Int
    nu::Int
    S::MT
    lujac::Fac
    Gu::SMT
end
function DirectSensitivity(lujac, Gu)
    nx, nu = size(Gu)
    S = zeros(nx, nu)
    return DirectSensitivity(nx, nu, S, lujac, Gu)
end

function update_op!(S::DirectSensitivity)
    copyto!(S.S, S.Gu)
    ldiv!(S.lujac, S.S)
end

function LinearAlgebra.mul!(y::AbstractArray, S::DirectSensitivity, x::AbstractArray, alpha, beta)
    mul!(y, S.S, x, -alpha, beta)
end

function tmul!(y::AbstractArray, S::DirectSensitivity, x::AbstractArray, alpha, beta)
    mul!(y, S.S', x, -alpha, beta)
end


#=
    Reduction algorithm
=#
abstract type AbstractReduction end

update!(red::AbstractReduction) = update_op!(red.S)

function tgtmul!(yx::AbstractArray, yu::AbstractArray, A::SparseMatrixCSC, z::AbstractArray, w::AbstractArray, alpha::Number, beta::Number)
    n, m = size(A)
    p = size(yx, 2)
    nz, nw = size(z, 1), size(w, 1)
    nx, nu = size(yx, 1), size(yu, 1)
    @assert m == nz + nw
    @assert n == nx + nu
    @assert p == size(yu, 2) == size(z, 2) == size(w, 2)

    yx .*= beta
    yu .*= beta

    @inbounds for j in 1:m
        for c in A.colptr[j]:A.colptr[j+1]-1
            i = A.rowval[c]
            for k in 1:p
                x = (j <= nz) ? z[j, k] : w[j - nz, k]
                if i <= nx
                    yx[i, k] += alpha * A.nzval[c] * x
                else
                    yu[i - nx, k] += alpha * A.nzval[c] * x
                end
            end
        end
    end
end

function tgtmul!(y::AbstractArray, A::SparseMatrixCSC, z::AbstractArray, w::AbstractArray, alpha::Number, beta::Number)
    n, m = size(A)
    nz, nw = size(z, 1), size(w, 1)
    p = size(z, 2)
    @assert m == nz + nw
    @assert p == size(z, 2) == size(w, 2)

    fill!(y, 0.0)

    @inbounds for j in 1:m
        for c in A.colptr[j]:A.colptr[j+1]-1
            i = A.rowval[c]
            for k in 1:p
                x = (j <= nz) ? z[j, k] : w[j - nz, k]
                y[i, k] += alpha * A.nzval[c] * x
            end
        end
    end
end

function direct_reduction!(red::AbstractReduction, y, J, Gu, w)
    @assert size(w, 2) == n_batches(red)
    # Load variables
    nbatch = n_batches(red)
    z = red.z

    mul!(z, red.S, w, 1.0, 0.0)
    tgtmul!(y, J, z, w, 1.0, 0.0)
    return
end

#=
    Adjoint-adjoint reduction
=#
function adjoint_adjoint_reduction!(red::AbstractReduction, hessvec, H, w)
    @assert size(w, 2) == n_batches(red)
    # Load variables
    nbatch = n_batches(red)
    z = red.z
    ψ = red.ψ

    mul!(z, red.S, w, 1.0, 0.0)
    tgtmul!(ψ, hessvec, H, z, w, 1.0, 0.0)
    tmul!(hessvec, red.S, ψ, 1.0, 1.0)
    return
end

struct Reduction{VT,Op} <: AbstractReduction
    z::VT
    ψ::VT
    S::Op
end

function Reduction(polar::PolarForm{T, VI, VT, MT}, S::SensitivityOperator, nx, nu) where {T, VI, VT, MT}
    z = VT(undef, nx) ; fill!(z, zero(T))
    ψ = VT(undef, nx) ; fill!(ψ, zero(T))
    return Reduction(z, ψ, S)
end
n_batches(hlag::Reduction) = 1

function reduce!(red::Reduction{VT}, dest, W) where VT
    nu = size(dest, 1)
    v_cpu = zeros(nu)
    v = VT(undef, nu)
    @inbounds for i in 1:nu
        fill!(v_cpu, 0)
        v_cpu[i] = 1.0
        copyto!(v, v_cpu)
        hv = @view dest[:, i]
        adjoint_adjoint_reduction!(red, hv, W, v)
    end
end


struct BatchReduction{MT,Op} <: AbstractReduction
    nbatch::Int
    z::MT
    ψ::MT
    tangents::MT
    S::Op
end

function BatchReduction(polar::PolarForm{T, VI, VT, MT}, S, nx, nu, nbatch) where {T, VI, VT, MT}
    z   = MT(undef, nx, nbatch) ; fill!(z, zero(T))
    ψ   = MT(undef, nx, nbatch) ; fill!(ψ, zero(T))
    v  = MT(undef, nu, nbatch)  ; fill!(v, zero(T))
    return BatchReduction(nbatch, z, ψ, v, S)
end
n_batches(hlag::BatchReduction) = hlag.nbatch

function reduce!(red::BatchReduction, dest, W)
    @assert n_batches(red) > 1
    n = size(dest, 1)
    nbatch = n_batches(red)

    # Allocate memory for tangents
    v = red.tangents

    N = div(n, nbatch, RoundDown)
    for i in 1:N
        # Init tangents on CPU
        offset = (i-1) * nbatch
        set_batch_tangents!(v, offset, n, nbatch)
        # Contiguous views!
        hm = @view dest[:, nbatch * (i-1) + 1: nbatch * i]
        adjoint_adjoint_reduction!(red, hm, W, v)
    end

    # Last slice
    last_batch = n - N*nbatch
    if last_batch > 0
        offset = n - nbatch
        set_batch_tangents!(v, offset, n, nbatch)

        hm = @view dest[:, (n - nbatch + 1) : n]
        adjoint_adjoint_reduction!(red, hm, W, v)
    end
end
