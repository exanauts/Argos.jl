
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

# TODO: forgiving function should not belong to Argos
function MadNLP.solve!(M::MadNLP.LapackCPUSolver{T}, x::SubArray{T}) where T
    if M.opt.lapack_algorithm == MadNLP.BUNCHKAUFMAN
        MadNLP.solve_bunchkaufman!(M,x)
    elseif M.opt.lapack_algorithm == MadNLP.LU
        MadNLP.solve_lu!(M,x)
    elseif M.opt.lapack_algorithm == MadNLP.QR
        MadNLP.solve_qr!(M,x)
    elseif M.opt.lapack_algorithm == MadNLP.CHOLESKY
        MadNLP.solve_cholesky!(M,x)
    else
        error(MadNLP.LOGGER,"Invalid lapack_algorithm")
    end
end

