
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
