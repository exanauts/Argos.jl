
function copy_index!(dest, src, idx)
    @assert length(dest) == length(idx)
    for i in eachindex(idx)
        dest[i] = src[idx[i]]
    end
end

# Split Jacobian in three parts Gx, Gu, A and return mapping
function _split_jacobian_csc(Ai, Ap, n, nx, nu)
    m = nx + nu # number of columns
    # First step: count
    nnzGx = 0
    nnzGu = 0
    nnzA = 0
    for j in 1:m
        for c in Ap[j]:Ap[j+1]-1
            i = Ai[c]
            if i <= nx && j <= nx
                nnzGx += 1
            elseif i <= nx && nx + 1 <= j <= nx + nu
                nnzGu += 1
            elseif nx + 1 <= i <= n
                nnzA += 1
            end
        end
    end

    mapGx = zeros(Int, nnzGx)
    mapGu = zeros(Int, nnzGu)
    mapA = zeros(Int, nnzA)

    k = 1
    lGx, lGu, lA = 1, 1, 1
    for j in 1:m
        for c in Ap[j]:Ap[j+1]-1
            i = Ai[c]
            if i <= nx && j <= nx
                mapGx[lGx] = k
                lGx += 1
            elseif i <= nx && nx + 1 <= j <= nx + nu
                mapGu[lGu] = k
                lGu += 1
            elseif nx + 1 <= i <= n
                mapA[lA] = k
                lA += 1
            end
            k += 1
        end
    end

    return mapA, mapGx, mapGu
end

function _split_jacobian_csr(Ap, Aj, n, nx, nu)
    m = nx + nu # number of columns
    # First step: count
    nnzGx = 0
    nnzGu = 0
    nnzA = 0
    for i in 1:n
        for c in Ap[i]:Ap[i+1]-1
            j = Aj[c]
            if i <= nx && j <= nx
                nnzGx += 1
            elseif i <= nx && nx + 1 <= j <= nx + nu
                nnzGu += 1
            elseif nx + 1 <= i <= n
                nnzA += 1
            end
        end
    end

    mapGx = zeros(Int, nnzGx)
    mapGu = zeros(Int, nnzGu)
    mapA = zeros(Int, nnzA)

    k = 1
    lGx, lGu, lA = 1, 1, 1
    for i in 1:n
        for c in Ap[i]:Ap[i+1]-1
            j = Aj[c]
            if i <= nx && j <= nx
                mapGx[lGx] = k
                lGx += 1
            elseif i <= nx && nx + 1 <= j <= nx + nu
                mapGu[lGu] = k
                lGu += 1
            elseif nx + 1 <= i <= n
                mapA[lA] = k
                lA += 1
            end
            k += 1
        end
    end

    return mapA, mapGx, mapGu
end

function split_jacobian(J::SparseMatrixCSC, nx, nu)
    n, m = size(J)
    @assert m == nx + nu
    return _split_jacobian_csc(J.rowval, J.colptr, n, nx, nu)
end

function _get_fixed_index_csc(n, m, Jp, Ji, ind_fixed, diag_ind)
    nnz_fixed = Int[]
    diag_ind_fixed = Int[]
    for j in ind_fixed
        for c in Jp[j]:Jp[j+1]-1
            if diag_ind && (Ji[c] == j)
                push!(diag_ind_fixed, c)
            else
                push!(nnz_fixed, c)
            end
        end
    end
    return nnz_fixed, diag_ind_fixed
end

function _get_fixed_index_csr(n, m, Jp, Jj, ind_fixed, diag_ind)
    nnz_fixed = Int[]
    diag_ind_fixed = Int[]
    for i in 1:n
        for c in Jp[i]:Jp[i+1]-1
            is_fixed = (Jj[c] in ind_fixed)
            if is_fixed
                if diag_ind && (Jj[c] == i)
                    push!(diag_ind_fixed, c)
                else
                    push!(nnz_fixed, c)
                end
            end
        end
    end
    return nnz_fixed, diag_ind_fixed
end

function get_fixed_nnz(J::SparseMatrixCSC, ind_fixed, diag_ind)
    n, m = size(J)
    Jp, Ji = J.colptr, J.rowval
    return _get_fixed_index_csc(n, m, Jp, Ji, ind_fixed, diag_ind)
end


struct HJDJ{VI, VT, SMT}
    W::SMT
    Jt::SMT
    JtJ::SMT
    Σ::VT
    transperm::VI
end
function HJDJ(W::SparseMatrixCSC, J::SparseMatrixCSC)
    n = size(W, 1)
    Jt = spzeros(0, 0)
    transperm = zeros(Int, 0)
    JtJ = J' * J
    constants = similar(nonzeros(W), n) ; fill!(constants, 0)
    return HJDJ(W, Jt, JtJ, constants, transperm)
end

function update!(K::HJDJ, A::SparseMatrixCSC, D::AbstractVector, Σ::AbstractVector)
    K.JtJ .= A' * Diagonal(D) * A
    K.Σ .= Σ
end

function LinearAlgebra.mul!(y::AbstractArray, K::HJDJ, x::AbstractArray)
    y .= K.Σ .* x
    mul!(y, K.W, x, 1.0, 1.0)
    mul!(y, K.JtJ, x, 1.0, 1.0)
end

function tgtmul!(yx::AbstractArray, yu::AbstractArray, K::HJDJ, z::AbstractArray, w::AbstractArray, alpha::Number, beta::Number)
    nx, nu = size(yx, 1), size(yu, 1)

    sx = view(K.Σ, 1:nx)
    su = view(K.Σ, 1+nx:nx+nu)
    yx .= sx .* z
    yu .= su .* w

    tgtmul!(yx, yu, K.JtJ, z, w, 1.0, 1.0)
    tgtmul!(yx, yu, K.W, z, w, 1.0, 1.0)
end

#=
    Special kernels for fixed variables.
=#

function fixed!(dest, ind_fixed, val)
    @inbounds for i in ind_fixed
        dest[i] = val
    end
end

function fixed_diag!(dest, ind_fixed, val)
    @inbounds for i in ind_fixed
        dest[i, i] = val
    end
end

