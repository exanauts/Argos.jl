# Taken from
# https://github.com/scipy/scipy/blob/3b36a574dc657d1ca116f6e230be694f3de31afc/scipy/sparse/sparsetools/csr.h#L376
function csr2csc(n, m, Ap, Aj, Ax, Bp, Bi, Bx)
    nnzA = Ap[n+1] - 1
    fill!(Bp, 0)

    for i in 1:nnzA
        Bp[Aj[i]] += 1
    end

    cumsum = 1
    for j in 1:m
        tmp = Bp[j]
        Bp[j] = cumsum
        cumsum += tmp
    end
    Bp[m+1] = nnzA + 1

    for i in 1:n
        for c in Ap[i]:Ap[i+1]-1
            j = Aj[c]
            dest = Bp[j]
            Bi[dest] = i
            Bx[dest] = Ax[c]
            Bp[j] += 1
        end
    end

    last = 1
    for j in 1:m+1
        tmp = Bp[j]
        Bp[j] = last
        last = tmp
    end
end

csc2csr(n, m, Ap, Ai, Ax, Bp, Bj, Bx) = csr2csc(m, n, Ap, Ai, Ax, Bp, Bj, Bx)

function decrement!(vals::Vector{Cint})
    vals .-= Cint(1)
    return
end

function convert2csr(A::SparseMatrixCSC{Tv, Ti}) where {Tv, Ti}
    n, m = size(A)
    nnzA = nnz(A)
    Ap, Ai, Ax = A.colptr, A.rowval, A.nzval

    Bp = zeros(Ti, n+1)
    Bj = zeros(Ti, nnzA)
    Bx = zeros(Tv, nnzA)

    csc2csr(n, m, Ap, Ai, Ax, Bp, Bj, Bx)
    return Bp, Bj, Bx
end

function drop_diag_csr(Bp::Vector{Ti}, Bj::Vector{Ti}, Bx::Vector{Tv}) where {Ti, Tv}
    n = length(Bp) - 1
    nnzB = Bp[n+1] - 1

    nnzC = nnzB - n
    Cp = zeros(Ti, n+1)
    Cj = zeros(Ti, nnzC)
    Cx = zeros(Tv, nnzC)

    k = 1
    for i in 1:n
        for c in Bp[i]:Bp[i+1]-1
            j = Bj[c]
            if i != j
                Cp[i] += 1
                Cj[k] = j
                Cx[k] = Bx[c]
                k += 1
            end
        end
    end
    last = 1
    for j in 1:n+1
        tmp = Cp[j]
        Cp[j] = last
        last += tmp
    end

    return Cp, Cj, Cx
end


