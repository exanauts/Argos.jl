
#=
    CondensedKKT
=#
function Argos.HJDJ(W::CuSparseMatrixCSR, J::CuSparseMatrixCSR)
    n = size(W, 1)

    # Perform initial computation on the CPU
    J_h = SparseMatrixCSC(J)
    Jt_h = SparseMatrixCSC(J_h')
    JtJ_h = Jt_h * J_h

    # Compute tranpose and associated permutation with CSC matrix
    i, j, _ = findnz(J_h)
    transperm = sortperm(i) |> CuVector{Int}
    # Initial transpose on GPU
    Jt = CuSparseMatrixCSR(Jt_h)
    # JDJ
    JtJ = CuSparseMatrixCSR(JtJ_h)

    constants = similar(nonzeros(W), n) ; fill!(constants, 0)
    return Argos.HJDJ{CuVector{Int}, CuVector{Float64}, typeof(W)}(W, Jt, JtJ, constants, transperm)
end

function Argos.update!(K::Argos.HJDJ, A, D, Σ)
    m = size(A, 1)
    _scale_transpose_kernel!(CUDABackend())(
        K.Jt.nzVal, A.rowPtr, A.colVal, A.nzVal, D, K.transperm,
        ndrange=(m, 1),
    )
    KA.synchronize(CUDABackend())
    spgemm!('N', 'N', 1.0, K.Jt, A, 0.0, K.JtJ, 'O')
    K.Σ .= Σ
end
function Argos.update!(K::Argos.HJDJ, A, D)
    m = size(A, 1)
    _scale_transpose_kernel!(CUDABackend())(
        K.Jt.nzVal, A.rowPtr, A.colVal, A.nzVal, D, K.transperm,
        ndrange=(m, 1),
    )
    KA.synchronize(CUDABackend())
    spgemm!('N', 'N', 1.0, K.Jt, A, 0.0, K.JtJ, 'O')
end

function MadNLP.set_aug_diagonal!(kkt::Argos.BieglerKKTSystem{T, VI, VT, MT}, solver::MadNLP.MadNLPSolver{T, Vector{T}}) where {T, VI<:CuVector{Int}, VT<:CuVector{T}, MT<:CuMatrix{T}}
    haskey(kkt.etc, :pr_diag_host) || (kkt.etc[:pr_diag_host] = Vector{T}(undef, length(kkt.pr_diag)))
    pr_diag_h = kkt.etc[:pr_diag_host]::Vector{T}
    # Broadcast is not working as MadNLP array are allocated on the CPU,
    # whereas pr_diag is allocated on the GPU
    x = MadNLP.full(solver.x)
    xl = MadNLP.full(solver.xl)
    xu = MadNLP.full(solver.xu)
    zl = MadNLP.full(solver.zl)
    zu = MadNLP.full(solver.zu)

    fill!(kkt.reg, 0.0)
    fill!(kkt.du_diag, 0.0)

    kkt.l_diag .= solver.xl_r .- solver.x_lr
    kkt.u_diag .= solver.x_ur .- solver.xu_r
    copyto!(kkt.l_lower, solver.zl_r)
    copyto!(kkt.u_lower, solver.zu_r)
    copyto!(pr_diag_h, kkt.reg)
    pr_diag_h[kkt.ind_lb] .-= kkt.l_lower ./ kkt.l_diag
    pr_diag_h[kkt.ind_ub] .-= kkt.u_lower ./ kkt.u_diag

    copyto!(kkt.pr_diag, pr_diag_h)
end
