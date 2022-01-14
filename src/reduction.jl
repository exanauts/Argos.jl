
# Small utils to compute the factorization for batch Hessian algorithm
function batch_factorization(J::AbstractSparseMatrix, nbatch)
    lufac = lu(J)
    if isnothing(lufac)
        error("Unable to find a factorization routine for type $(typeof(J))")
    end
    return (lufac, lufac')
end

abstract type AbstractReduction end

## Auxiliary functions
function init_tangent!(tgt::AbstractArray, z::AbstractArray, w::AbstractArray, nx, nu)
    nbatch = size(tgt, 2)
    for i in 1:nbatch
        for j in 1:nx
            @inbounds tgt[j, i] = z[j, i]
        end
        for j in 1:nu
            @inbounds tgt[j+nx, i] = w[j, i]
        end
    end
end

function split_array!(src::AbstractVector, dfx::AbstractVector, dfu::AbstractVector, nx, nu)
    dfx .= @view src[1:nx]
    dfu .= @view src[nx+1:nx+nu]
    return
end
function split_array!(src::AbstractMatrix, dfx::AbstractMatrix, dfu::AbstractMatrix, nx, nu)
    dfx .= @view src[1:nx, :]
    dfu .= @view src[nx+1:nx+nu, :]
    return
end

#=
    Direct reduction
    Operator [I ; -inv(Gx) * Gu]
=#
function direct_reduction!(red::AbstractReduction, jvec, J, Gu, w)
    @assert size(w, 2) == n_batches(red)
    # Load variables
    nx, nu = red.nx, red.nu
    nbatch = n_batches(red)
    tgt = red.tmp_tgt
    z = red.z

    # Step 1: solve linear system inv(Gx) * Gu * w
    mul!(z, Gu, w, -1.0, 0.0)
    LinearAlgebra.ldiv!(red.lu, z)

    init_tangent!(tgt, z, w, nx, nu)
    mul!(jvec, J, tgt)
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
    tgt = red.tmp_tgt
    hv = red.tmp_hv
    z = red.z
    ψ = red.ψ

    # Step 1: computation of first second-order adjoint
    mul!(z, Gu, w, -1.0, 0.0)
    LinearAlgebra.ldiv!(red.lu, z)

    # STEP 2: AutoDiff
    # Init tangent with z and w
    init_tangent!(tgt, z, w, nx, nu)
    # SpMV
    mul!(hv, H, tgt)
    # Get results
    split_array!(hv, ψ, hessvec, nx, nu)

    # STEP 3: computation of second second-order adjoint
    LinearAlgebra.ldiv!(red.adjlu, ψ)
    mul!(hessvec, transpose(Gu), ψ, -1.0, 1.0)

    return
end


struct Reduction{VT,Fac1,Fac2} <: AbstractReduction
    nx::Int
    nu::Int
    # Adjoints
    z::VT
    ψ::VT
    # Buffers
    _w1::VT
    # Tangents
    tmp_tgt::VT
    tmp_hv::VT
    lu::Fac1
    adjlu::Fac2
end

function Reduction(polar::PolarForm{T, VI, VT, MT}, J::AbstractSparseMatrix, ncons::Int) where {T, VI, VT, MT}
    lu1, lu2 = batch_factorization(J, 1)
    nx, nu = ExaPF.number(polar, ExaPF.State()), ExaPF.number(polar, ExaPF.Control())
    z = VT(undef, nx)
    ψ = VT(undef, nx)
    tgt = VT(undef, nx+nu)
    hv = VT(undef, nx+nu)
    _w1 = VT(undef, ncons)
    return Reduction(nx, nu, z, ψ, _w1, tgt, hv, lu1, lu2)
end
n_batches(hlag::Reduction) = 1



struct BatchReduction{MT,Fac1,Fac2} <: AbstractReduction
    nx::Int
    nu::Int
    nbatch::Int
    # Adjoints
    z::MT
    ψ::MT
    # Buffer
    _w1::MT
    _w2::MT
    _w3::MT
    # Tangents
    tangents::MT
    tmp_tgt::MT
    tmp_hv::MT
    lu::Fac1
    adjlu::Fac2
end

function BatchReduction(polar::PolarForm{T, VI, VT, MT}, J, nbatch, ncons) where {T, VI, VT, MT}
    lu1, lu2 = _batch_hessian_factorization(J, nbatch)
    nx, nu = ExaPF.get(polar, ExaPF.NumberOfState()), ExaPF.get(polar, ExaPF.NumberOfControl())
    m = ExaPF.size_constraint(polar, func)::Int
    z   = MT(undef, nx, nbatch)
    ψ   = MT(undef, nx, nbatch)
    tgt = MT(undef, nx+nu, nbatch)
    hv  = MT(undef, nx+nu, nbatch)
    v  = MT(undef, nu, nbatch)
    _w1 = MT(undef, ncons, nbatch)
    _w2 = MT(undef, nx, nbatch)
    _w3 = MT(undef, nu, nbatch)
    return BatchReduction(nx, nu, nbatch, z, ψ, _w1, _w2, _w3, v, tgt, hv, lu1, lu2)
end
n_batches(hlag::BatchReduction) = hlag.nbatch

function update_factorization!(hlag::AbstractReduction, J::AbstractSparseMatrix)
    LinearAlgebra.lu!(hlag.lu, J)
    return
end
