
abstract type AbstractReduction end

function tgtmul!(yx::AbstractArray, yu::AbstractArray, A::SparseMatrixCSC, z::AbstractArray, w::AbstractArray)
    n, m = size(A)
    p = size(yx, 2)
    nz, nw = size(z, 1), size(w, 1)
    nx, nu = size(yx, 1), size(yu, 1)
    @assert m == nz + nw
    @assert n == nx + nu
    @assert p == size(yu, 2) == size(z, 2) == size(w, 2)

    fill!(yx, 0.0)
    fill!(yu, 0.0)

    @inbounds for j in 1:m
        for c in A.colptr[j]:A.colptr[j+1]-1
            i = A.rowval[c]
            for k in 1:p
                x = (j <= nz) ? z[j, k] : w[j - nz, k]
                if i <= nx
                    yx[i, k] += A.nzval[c] * x
                else
                    yu[i - nx, k] += A.nzval[c] * x
                end
            end
        end
    end
end

function tgtmul!(y::AbstractArray, A::SparseMatrixCSC, z::AbstractArray, w::AbstractArray)
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
                y[i, k] += A.nzval[c] * x
            end
        end
    end
end

function direct_reduction!(red::AbstractReduction, y, J, Gu, w)
    @assert size(w, 2) == n_batches(red)
    # Load variables
    nx, nu = red.nx, red.nu
    nbatch = n_batches(red)
    z = red.z

    mul!(z, Gu, w, -1.0, 0.0)
    LinearAlgebra.ldiv!(red.lujac, z)
    tgtmul!(y, J, z, w)
    return
end

#=
    Adjoint-adjoint reduction
=#
function adjoint_adjoint_reduction!(red::AbstractReduction, hessvec, H, Gu, w)
    @assert size(w, 2) == n_batches(red)
    # Load variables
    nx, nu = red.nx, red.nu
    nbatch = n_batches(red)
    z = red.z
    ψ = red.ψ

    # Step 1: computation of first second-order adjoint
    mul!(z, Gu, w, -1.0, 0.0)
    LinearAlgebra.ldiv!(red.lujac, z)

    # STEP 2: mul
    # SpMV
    tgtmul!(ψ, hessvec, H, z, w)

    # STEP 3: computation of second second-order adjoint
    LinearAlgebra.ldiv!(red.lujac', ψ)
    mul!(hessvec, transpose(Gu), ψ, -1.0, 1.0)
    return
end

struct Reduction{VT} <: AbstractReduction
    nx::Int
    nu::Int
    # Adjoints
    z::VT
    ψ::VT
    lujac::LinearAlgebra.Factorization
end

function Reduction(polar::PolarForm{T, VI, VT, MT}, lujac::Factorization) where {T, VI, VT, MT}
    nx, nu = ExaPF.number(polar, ExaPF.State()), ExaPF.number(polar, ExaPF.Control())
    z = VT(undef, nx)
    ψ = VT(undef, nx)
    return Reduction(nx, nu, z, ψ, lujac)
end
n_batches(hlag::Reduction) = 1


struct BatchReduction{MT} <: AbstractReduction
    nx::Int
    nu::Int
    nbatch::Int
    # Adjoints
    z::MT
    ψ::MT
    # Tangents
    tangents::MT
    lujac::LinearAlgebra.Factorization
end

function BatchReduction(polar::PolarForm{T, VI, VT, MT}, lujac, nbatch) where {T, VI, VT, MT}
    nx, nu = ExaPF.number(polar, ExaPF.State()), ExaPF.number(polar, ExaPF.Control())
    z   = MT(undef, nx, nbatch)
    ψ   = MT(undef, nx, nbatch)
    v  = MT(undef, nu, nbatch)
    return BatchReduction(nx, nu, nbatch, z, ψ, v, lujac)
end
n_batches(hlag::BatchReduction) = hlag.nbatch

