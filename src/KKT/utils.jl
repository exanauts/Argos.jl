
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

    tgtmul!(yx, yu, K.W, z, w, 1.0, 1.0)
    tgtmul!(yx, yu, K.JtJ, z, w, 1.0, 1.0)
end

