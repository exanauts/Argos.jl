@kernel function _map2vec_kernel!(dest, src, map)
    i = @index(Global, Linear)
    dest[i] = src[map[i]]
end
function Argos.transfer2tril!(hessvals::AbstractVector, H::CuSparseMatrixCSR, csc2tril)
    Hz = nonzeros(H)
    ndrange = (length(hessvals),)
    _map2vec_kernel!(CUDABackend())(hessvals, Hz, csc2tril; ndrange=ndrange)
    KA.synchronize(CUDABackend())
end


@kernel function _fixed_kernel!(dest, fixed, val)
    i = @index(Global, Linear)
    dest[fixed[i]] = val
end
function Argos.fixed!(dest::CuVector, ind_fixed, val::Number)
    length(ind_fixed) == 0 && return
    g_ind_fixed = ind_fixed |> CuArray
    _fixed_kernel!(CUDABackend())(dest, g_ind_fixed, val; ndrange=length(ind_fixed))
    KA.synchronize(CUDABackend())
end


@kernel function _copy_index_kernel!(dest, src, idx)
    i = @index(Global, Linear)
    @inbounds dest[i] = src[idx[i]]
end
function Argos.copy_index!(dest::CuVector{T}, src::CuVector{T}, idx) where T
    @assert length(dest) == length(idx)
    ndrange = (length(dest),)
    idx_d = CuVector(idx)
    _copy_index_kernel!(CUDABackend())(dest, src, idx_d; ndrange=ndrange)
    KA.synchronize(CUDABackend())
end


@kernel function _fixed_diag_kernel!(dest, fixed, val)
    i = @index(Global, Linear)
    k = fixed[i]
    dest[k, k] = val
end
function Argos.fixed_diag!(dest::CuMatrix, ind_fixed, val::Number)
    length(ind_fixed) == 0 && return
    g_ind_fixed = ind_fixed |> CuArray
    _fixed_diag_kernel!(CUDABackend())(dest, g_ind_fixed, val; ndrange=length(ind_fixed))
    KA.synchronize(CUDABackend())
end

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
    _transfer_auglag_hessian!(CUDABackend())(dest, H, J, ρ, n, m, ndrange=ndrange)
    KA.synchronize(CUDABackend())
    return
end


@kernel function _batch_tangents_kernel!(seeds, offset, n_batches)
    i = @index(Global, Linear)
    @inbounds seeds[i + offset, i] = 1.0
end
function Argos.set_batch_tangents!(seeds::CuMatrix, offset, n, n_batches)
    @assert offset + n_batches <= n
    ndrange = (n_batches)
    fill!(seeds, 0.0)
    _batch_tangents_kernel!(CUDABackend())(
        seeds, offset, n_batches;
        ndrange=ndrange,
    )
    KA.synchronize(CUDABackend())
    return
end


@kernel function _tgtmul_1_kernel!(y, A_rowPtr, A_colVal, A_nzVal, z, w, alpha, nx, nu)
    i, k = @index(Global, NTuple)
    @inbounds for c in A_rowPtr[i]:A_rowPtr[i+1]-1
        j = A_colVal[c]
        if j <= nx
            @inbounds y[i, k] += alpha * A_nzVal[c] * z[j, k]
        else
            @inbounds y[i, k] += alpha * A_nzVal[c] * w[j - nx, k]
        end
    end
end
function Argos.tgtmul!(
    y::AbstractArray, A::CuSparseMatrixCSR, z::AbstractArray, w::AbstractArray,
    alpha::Number, beta::Number,
)
    n, m = size(A)
    nz, nw = size(z, 1), size(w, 1)
    @assert m == nz + nw
    @assert size(z, 2) == size(w, 2) == size(y, 2)
    k = size(z, 2)
    ndrange = (n, k)
    y .*= beta
    _tgtmul_1_kernel!(CUDABackend())(
        y, A.rowPtr, A.colVal, A.nzVal, z, w, alpha, nz, nw;
        ndrange=ndrange,
    )
    KA.synchronize(CUDABackend())
end


@kernel function _tgtmul_2_kernel!(yx, yu, A_rowPtr, A_colVal, A_nzVal, z, w, alpha, nx, nu)
    i, k = @index(Global, NTuple)
    @inbounds for c in A_rowPtr[i]:A_rowPtr[i+1]-1
        j = A_colVal[c]
        if (i <= nx) && (j <= nx)
            @inbounds yx[i, k] += alpha * A_nzVal[c] * z[j, k]
        elseif (i <= nx) && (j <= nx + nu)
            @inbounds yx[i, k] += alpha * A_nzVal[c] * w[j - nx, k]
        elseif (i <= nx + nu) && (j <= nx)
            @inbounds yu[i - nx, k] += alpha * A_nzVal[c] * z[j, k]
        elseif (i <= nx + nu) && (j <= nx + nu)
            @inbounds yu[i - nx, k] += alpha * A_nzVal[c] * w[j - nx, k]
        end
    end
end
function Argos.tgtmul!(
    yx::AbstractArray, yu::AbstractArray, A::CuSparseMatrixCSR, z::AbstractArray, w::AbstractArray,
    alpha::Number, beta::Number,
)
    n, m = size(A)
    nz, nw = size(z, 1), size(w, 1)
    @assert m == nz + nw
    @assert size(z, 2) == size(w, 2) == size(yx, 2) == size(yu, 2)
    k = size(z, 2)
    ndrange = (n, k)
    yx .*= beta
    yu .*= beta
    _tgtmul_2_kernel!(CUDABackend())(
        yx, yu, A.rowPtr, A.colVal, A.nzVal, z, w, alpha, nz, nw;
        ndrange=ndrange,
    )
    KA.synchronize(CUDABackend())
end


@kernel function _scale_transpose_kernel!(
    Jtz, Jp, Jj, Jz, D, tperm,
)
    i = @index(Global, Linear)

    for c in Jp[i]:Jp[i+1]-1
        j = Jj[c]
        Jtz[tperm[c]] = D[i] * Jz[c]
    end
end

function Argos.update!(K::Argos.HJDJ, A, D, Σ)
    m = size(A, 1)
    ev = _scale_transpose_kernel!(CUDABackend())(
        K.Jt.nzVal, A.rowPtr, A.colVal, A.nzVal, D, K.transperm,
        ndrange=(m, 1),
    )
    KA.synchronize(ev)
    spgemm!('N', 'N', 1.0, K.Jt, A, 0.0, K.JtJ, 'O')
    K.Σ .= Σ
end
