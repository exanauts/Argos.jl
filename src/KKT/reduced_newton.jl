
"""
    BieglerKKTSystem{T, VI, VT, MT, SMT} <: MadNLP.AbstractReducedKKTSystem{T, VT, MT}

Implementation of Biegler's reduction method [BNS2015] in MadNLP's syntax.
The API follows the [MadNLP's specifications](https://madnlp.github.io/MadNLP.jl/dev/lib/kkt/#KKT-systems). The reduction is at the basis of the *linearize-then-reduce*
method described in [PSSMA2022].

Return a dense matrix that can be factorized efficiently inside MadNLP by any dense
linear algebra routine (e.g. Lapack).

## Examples

```julia-repl
julia> flp = Argos.FullSpaceEvaluator("case9.m")

julia> opf = Argos.OPFModel(flp)

julia> T = Float64

julia> VI, VT, MT = Vector{Int}, Vector{T}, Matrix{T}

julia> kkt = Argos.BieglerKKTSystem{T, VI, VT, MT}(opf)

julia> MadNLP.get_kkt(kkt) # return the matrix to factorize

```

## Notes

`BieglerKKTSystem` can be instantiated both on the host memory (CPU)
or on a NVIDIA GPU using CUDA. When instantiated on the GPU, `BieglerKKTSystem`
uses `cusolverRF` to streamline the solution of the sparse linear systems
in the reduction algorithm.

## References

[BNS2015] Biegler, Lorenz T., Jorge Nocedal, and Claudia Schmid. "A reduced Hessian method for large-scale constrained optimization." SIAM Journal on Optimization 5, no. 2 (1995): 314-347.

[PSSMA2022] Pacaud, François, Sungho Shin, Michel Schanen, Daniel Adrian Maldonado, and Mihai Anitescu. "Condensed interior-point methods: porting reduced-space approaches on GPU hardware." arXiv preprint arXiv:2203.11875 (2022).

"""
struct BieglerKKTSystem{
    T,
    VI,
    VT,
    MT,
    SMT,
    LS,
} <: MadNLP.AbstractReducedKKTSystem{T, VT, MT, MadNLP.ExactHessian{T, VT}}
    K::HJDJ{VI,VT,SMT}
    Wref::SMT
    W::SMT
    J::SMT
    A::SMT
    Gx::SMT
    Gu::SMT
    mapA::VI
    mapGx::VI
    mapGu::VI
    # Hessian nzval
    h_V::VT
    # Jacobian nzval
    j_V::VT
    # Regularization terms
    reg::Vector{T}
    pr_diag::VT
    du_diag::Vector{T}
    l_diag::Vector{T}
    u_diag::Vector{T}
    l_lower::Vector{T}
    u_lower::Vector{T}
    # Reduced KKT system
    aug_com::MT
    reduction::AbstractReduction
    # Pivot
    G_fac::LinearAlgebra.Factorization
    # Buffers
    _wxu1::VT
    _wxu2::VT
    _wxu3::VT
    _wx1::VT
    _wx2::VT
    _wj1::VT
    _wj2::VT
    # Dimension
    nx::Int
    nu::Int
    # Info
    ind_ineq::Vector{Int}
    ind_lb::Vector{Int}
    ind_ub::Vector{Int}
    ind_fixed::Vector{Int}
    ind_Gu_fixed::VI
    ind_A_fixed::VI
    ind_W_fixed_diag::VI
    ind_W_fixed::VI
    jacobian_scaling::VT
    linear_solver::LS
    etc::Dict{Symbol,Any}
end

function MadNLP.create_kkt_system(
    ::Type{BieglerKKTSystem{T, VI, VT, MT}},
    cb::MadNLP.AbstractCallback{T, Vector{T}},
    opt, opt_linear_solver, cnt, ind_cons;
    max_batches=256,
) where {T, VI, VT, MT}
    nlp = cb.nlp
    n_slack = length(ind_cons.ind_ineq)
    n = NLPModels.get_nvar(nlp)
    m = NLPModels.get_ncon(nlp)
    nlb, nub = length(ind_cons.ind_lb), length(ind_cons.ind_ub)
    # Structure
    evaluator = backend(nlp)
    nx = evaluator.nx
    nu = evaluator.nu
    # Evaluate sparsity pattern
    nnzj = NLPModels.get_nnzj(nlp)
    nnzh = NLPModels.get_nnzh(nlp)

    Wref = evaluator.hess.H
    W = copy(Wref)
    SMT = typeof(W)
    h_V = VT(undef, nnzh) ; fill!(h_V, zero(T))

    J = copy(evaluator.jac.J)
    j_V = VT(undef, nnzj) ; fill!(j_V, zero(T))

    # Instantiate on host
    J_h = SparseMatrixCSC(J)
    W_h = SparseMatrixCSC(W)

    # Decompose J
    Gx_h = J_h[1:nx, 1:nx]
    Gu_h = J_h[1:nx, nx+1:nx+nu]
    A_h = J_h[nx+1:end, :]
    # Associated mappings
    mapA, mapGx, mapGu = split_jacobian(J, nx, nu)

    # Transfer to device
    Gx = Gx_h |> SMT
    Gu = Gu_h |> SMT
    A = A_h |> SMT

    # Condensed matrix
    K = HJDJ(W, A)

    pr_diag = VT(undef, n + n_slack) ; fill!(pr_diag, zero(T))
    du_diag = VT(undef, m) ; fill!(du_diag, zero(T))

    reg = zeros(n + n_slack)

    l_diag = zeros(nlb)
    u_diag = zeros(nub)
    l_lower = zeros(nlb)
    u_lower = zeros(nub)

    nbatches = min(max_batches, nu)
    linear_solver = LS.DirectSolver(Gx; nrhs=nbatches)
    Gxi = linear_solver.factorization
    S = ImplicitSensitivity(Gxi, Gu)
    reduction = if nbatches > 1
        BatchReduction(evaluator.model, S, nx, nu, nbatches)
    else
        Reduction(evaluator.model, S, nx, nu)
    end

    # W
    aug_com = MT(undef, nu, nu) ; fill!(aug_com, zero(T))

    # Buffers
    _wxu1 = VT(undef, nx+nu) ; fill!(_wxu1, 0)
    _wxu2 = VT(undef, nx+nu) ; fill!(_wxu2, 0)
    _wxu3 = VT(undef, nx+nu) ; fill!(_wxu3, 0)
    _wx1 = VT(undef, nx)     ; fill!(_wx1, 0)
    _wx2 = VT(undef, nx)     ; fill!(_wx2, 0)
    _wj1 = VT(undef, m-nx)   ; fill!(_wj1, 0)
    _wj2 = VT(undef, m-nx)   ; fill!(_wj2, 0)

    # Scaling
    jacobian_scaling = VT(undef, nnzj) ; fill!(jacobian_scaling, one(T))

    ind_fixed = ind_cons.ind_fixed
    # The states are not supposed to be fixed
    # Assume ind_fixed is sorted
    if (length(ind_fixed) > 0) && (ind_fixed[1] <= nx)
        error("Found a fixed state variable. Currently not supported as the Jacobian Gₓ becomes non-invertible.")
    end
    # Get fixed views
    ind_A_fixed, _ = get_fixed_nnz(A, ind_fixed, false)
    ind_Gu_fixed, _ = get_fixed_nnz(Gu, ind_fixed .- nx, false)
    ind_W_fixed, ind_W_fixed_diag = get_fixed_nnz(W, ind_fixed, true)

    cnt.linear_solver_time += @elapsed begin
        linear_solver = opt.linear_solver(aug_com; opt = opt_linear_solver)
    end
    # Buffers
    etc = Dict{Symbol, Any}(:reduction_time=>0.0, :cond=>Float64[], :scaling_initialized=>false)

    return BieglerKKTSystem{T, VI, VT, MT, SMT, typeof(linear_solver)}(
        K, Wref, W, J, A, Gx, Gu, mapA, mapGx, mapGu,
        h_V, j_V,
        reg, pr_diag, du_diag,
        l_diag, u_diag, l_lower, u_lower,
        aug_com, reduction, Gxi,
        _wxu1, _wxu2, _wxu3, _wx1, _wx2, _wj1, _wj2,
        nx, nu,
        ind_cons.ind_ineq, ind_cons.ind_lb, ind_cons.ind_ub, ind_fixed,
        ind_Gu_fixed, ind_A_fixed, ind_W_fixed_diag, ind_W_fixed,
        jacobian_scaling, linear_solver,
        etc,
    )
end

MadNLP.num_variables(kkt::BieglerKKTSystem) = kkt.nu
MadNLP.get_hessian(kkt::BieglerKKTSystem) = kkt.h_V
MadNLP.get_jacobian(kkt::BieglerKKTSystem) = nonzeros(kkt.J)
MadNLP.is_reduced(::BieglerKKTSystem) = true

# Return SparseMatrixCOO to MadNLP
function MadNLP.get_raw_jacobian(kkt::BieglerKKTSystem)
    m = size(kkt.J, 1)
    n = kkt.nx + kkt.nu
    i, j, v = findnz(kkt.J)
    return MadNLP.SparseMatrixCOO(m, n, i, j, v)
end

function MadNLP.initialize!(kkt::BieglerKKTSystem)
    fill!(kkt.reg, 1.0)
    fill!(kkt.pr_diag, 1.0)
    fill!(kkt.du_diag, 0.0)
    fill!(kkt.l_lower, 0.0)
    fill!(kkt.u_lower, 0.0)
    fill!(kkt.l_diag, 1.0)
    fill!(kkt.u_diag, 1.0)
    fill!(nonzeros(kkt.W), 0.0)
end

function _load_buffer(kkt::BieglerKKTSystem{T,VI,VT,MT}, x::AbstractVector, key::Symbol) where {T,VI,VT,MT}
    haskey(kkt.etc, key) || (kkt.etc[key] = VT(undef, length(x)))
    xb = kkt.etc[key]::VT
    # Consistency check
    @assert length(xb) == length(x)
    _copyto!(xb, 1, x, 1, length(x))
    return xb
end
function _load_buffer(kkt::BieglerKKTSystem{T,VI,VT,MT}, x::VT, key::Symbol) where {T,VI,VT,MT}
    return x
end

# Use for inertia-free regularization (require full-space multiplication)
function _mul_expanded!(y_h, kkt::BieglerKKTSystem{T, VI, VT, MT}, x_h, alpha, beta) where {T, VI, VT, MT}
    x = _load_buffer(kkt, x_h, :hess_x)::VT
    y = _load_buffer(kkt, y_h, :hess_y)::VT
    # Build full-space KKT system
    n = kkt.nx + kkt.nu
    m = size(kkt.J, 1)
    ns = length(kkt.ind_ineq)
    W = kkt.W
    Jx = kkt.J

    # Decompose x
    xx = view(x, 1:n)
    xs = view(x, n+1:n+ns)
    xz = view(x, n+ns+1:n+ns+m)
    # Decompose y
    yx = view(y, 1:n)
    ys = view(y, n+1:n+ns)
    yz = view(y, n+ns+1:n+ns+m)

    # / x (variable)
    mul!(yx, W, xx, alpha, beta)
    mul!(yx, Jx', xz, alpha, one(T))
    # / s (slack)
    ys .= beta .* ys .- alpha .* xz[kkt.nx+1:m]
    # / z (multipliers)
    mul!(yz, Jx, xx, alpha, beta)
    yz[kkt.nx+1:m] .-= alpha .* xs

    copyto!(y_h, y)
    return y_h
end

function MadNLP.mul!(y::VT, kkt::BieglerKKTSystem, x::VT, alpha, beta) where VT <: MadNLP.AbstractKKTVector
    _mul_expanded!(y.values, kkt, x.values, alpha, beta)
    MadNLP._kktmul!(y, x, kkt.reg, kkt.du_diag, kkt.l_lower, kkt.u_lower, kkt.l_diag, kkt.u_diag, alpha, beta)
    return y
end

# N.B.: custom wrapper for eval_jac_wrapper
#       to handle scaling of the Jacobian on the GPU.
function MadNLP.eval_jac_wrapper!(
    solver::MadNLP.MadNLPSolver,
    kkt::BieglerKKTSystem,
    x::MadNLP.PrimalVector{T},
) where T
    if !kkt.etc[:scaling_initialized]
        copyto!(kkt.jacobian_scaling, solver.cb.jac_scale)
    end
    jac = MadNLP.get_jacobian(kkt)
    solver.cnt.eval_function_time += @elapsed begin
        NLPModels.jac_coord!(solver.cb.nlp, MadNLP.variable(x), jac)
    end
    MadNLP.compress_jacobian!(kkt)
    solver.cnt.con_jac_cnt += 1
    return jac
end

function MadNLP.jtprod!(
    y_h::AbstractVector,
    kkt::BieglerKKTSystem{T, VI, VT, MT},
    x_h::AbstractVector,
) where {T, VI, VT, MT}
    x = _load_buffer(kkt, x_h, :jacx)::VT
    y = _load_buffer(kkt, y_h, :jacy)::VT
    n = kkt.nx + kkt.nu
    ns = length(kkt.ind_ineq)
    m = length(x)
    yx = view(y, 1:n)
    mul!(yx, kkt.J', x)
    ys = view(y, n+1:n+ns)
    xs = view(x, kkt.nx+1:m)
    ys .= .-xs

    copyto!(y_h, y)
end

MadNLP.nnz_jacobian(kkt::BieglerKKTSystem) = length(kkt.j_V)

function MadNLP.compress_jacobian!(kkt::BieglerKKTSystem)
    nx, nu = kkt.nx, kkt.nu
    Jv = nonzeros(kkt.J)
    Jv .*= kkt.jacobian_scaling
    # Build Jacobian
    copy_index!(nonzeros(kkt.Gx), Jv, kkt.mapGx)
    copy_index!(nonzeros(kkt.Gu), Jv, kkt.mapGu)
    copy_index!(nonzeros(kkt.A), Jv, kkt.mapA)

    Gxi = kkt.G_fac
    lu!(Gxi, kkt.Gx)

    fixed!(nonzeros(kkt.Gu), kkt.ind_Gu_fixed, 0.0)
    fixed!(nonzeros(kkt.A), kkt.ind_A_fixed, 0.0)
    return
end

# Build reduced Hessian
function MadNLP.compress_hessian!(kkt::BieglerKKTSystem)
    copyto!(nonzeros(kkt.W), nonzeros(kkt.Wref))
end

function assemble_condensed_matrix!(kkt::BieglerKKTSystem, K::HJDJ)
    nx, nu = kkt.nx, kkt.nu
    m = size(kkt.J, 1)
    ns = size(kkt.A, 1)
    D = kkt._wj1
    prx = @view kkt.pr_diag[1:nx+nu]
    prs = @view kkt.pr_diag[nx+nu+1:nx+nu+ns]
    Σd = @view kkt.du_diag[nx+1:m]
    # Matrices
    A = kkt.A
    D .= prs
    fixed!(prx, kkt.ind_fixed, 0.0)
    update!(K, A, D, prx)
end

function MadNLP.build_kkt!(kkt::BieglerKKTSystem{T, VI, VT, MT}) where {T, VI, VT, MT}
    nx = kkt.nx
    fixed!(nonzeros(kkt.W), kkt.ind_W_fixed, 0.0)
    fixed!(nonzeros(kkt.W), kkt.ind_W_fixed_diag, 1.0)

    assemble_condensed_matrix!(kkt, kkt.K)
    fill!(kkt.aug_com, 0.0)
    update!(kkt.reduction)
    timed = @elapsed begin
        reduce!(kkt.reduction, kkt.aug_com, kkt.K)
    end
    kkt.etc[:reduction_time] += timed

    # Regularize final reduced matrix to ensure it is full-rank
    fixed_diag!(kkt.aug_com, kkt.ind_fixed .- nx, 1.0)

    return
end

function MadNLP.solve!(
    kkt::BieglerKKTSystem{T, VI, VT, MT},
    w::MadNLP.AbstractKKTVector,
) where {T, VI, VT, MT}
    nv, m = size(kkt.W, 1), size(kkt.J, 1)
    nx, nu = kkt.nx, kkt.nu
    ns = m - nx
    n = nv + ns

    # Build reduced KKT vector.
    MadNLP.reduce_rhs!(w.xp_lr, MadNLP.dual_lb(w), kkt.l_diag, w.xp_ur, MadNLP.dual_ub(w), kkt.u_diag)
    # Intermediate buffer instantied on the device.
    x = _load_buffer(kkt, MadNLP.primal_dual(w), :kkt_x)::VT

    # Buffers
    jv = kkt._wxu1
    tx = view(jv, 1:nx)
    tu = view(jv, nx+1:nx+nu)
    vj = kkt._wj1
    sx1 = kkt._wx1
    sx2 = kkt._wx2
    # Views for Hessian-vector multiplication
    kv = kkt._wxu2
    kh = kkt._wxu3
    kvx = view(kv, 1:nx)
    kvu = view(kv, nx+1:nx+nu)
    khx = view(kh, 1:nx)
    khu = view(kh, nx+1:nx+nu)

    # Gₓ⁻¹
    Gxi = kkt.G_fac
    Gx = kkt.Gx
    Gu = kkt.Gu
    K = kkt.K
    Σₛ = view(kkt.pr_diag, nx+nu+1:nx+nu+ns)
    vs = kkt._wj2

    # Decompose solution
    dxu = view(x, 1:nx+nu)
    dx = view(x, 1:nx)           # / state
    du = view(x, 1+nx:nx+nu)     # / control
    ds = view(x, 1+nx+nu:n)      # / slack
    dλ = view(x, n+1:n+nx)       # / equality cons
    dy = view(x, n+nx+1:n+m)     # / inequality cons

    # Reduction (1) --- Condensed
    vj .= (Σₛ .* dy .+ ds)                # v = (Σₛ r₅ + α r₃)
    mul!(jv, kkt.A', vj, one(T), zero(T)) # jᵥ = Aᵀ v
    jv .+= dxu                            # r₁₂ - Aᵀv
    # Reduction (2) --- Biegler
    sx1 .= dλ                             # r₄
    ldiv!(Gxi, sx1)                       # Gₓ⁻¹ r₄

    sx2 .= tx                             # tx = jv[1:nx]
    kvx .= sx1 ; kvu .= zero(T)
    mul!(kh, K, kv)                       # [Kₓₓ Gₓ⁻¹ r₄ ; Kᵤₓ Gₓ⁻¹ r₄ ]
    sx2 .= khx .- tx                      # sₓ = Kₓₓ Gₓ⁻¹ r₄ - tₓ
    ldiv!(Gxi', sx2)                      # Gₓ⁻ᵀ sₓ
    mul!(tu, Gu', sx2, one(T), one(T))    # tᵤ = tᵤ + Gᵤᵀ Gₓ⁻ᵀ sₓ
    axpy!(-one(T), khu, tu)               # tᵤ = tᵤ - Kᵤₓ Gₓ⁻¹ r₄

    du .= tu
    MadNLP.solve!(kkt.linear_solver, du)

    # (1) Extract Biegler
    dx .= dλ                              # r₄
    mul!(dx, Gu, du, -one(T), one(T))     # r₄ - Gᵤ dᵤ
    ldiv!(Gxi, dx)                        # dₓ = Gₓ⁻¹ (r₄ - Gᵤ dᵤ)
    dλ .= tx                              # tₓ
    mul!(kh, K, dxu)                      # Kₓₓ dₓ + Kₓᵤ dᵤ
    axpy!(-one(T), khx, dλ)               # tₓ - Kₓₓ dₓ - Kₓᵤ dᵤ

    # TODO: SEGFAULT
    ldiv!(Gxi', dλ)                       # dₗ = Gₓ⁻ᵀ(tₓ - Kₓₓ dₓ + Kₓᵤ dᵤ)

    # (2) Extract Condensed
    mul!(vj, kkt.A, dxu)                  # Aₓ dₓ + Aᵤ dᵤ
    copyto!(vs, ds)
    ds .= (vj .- dy)
    dy .= Σₛ .* ds .- vs

    # Copy back primal-dual direction to MadNLP.
    copyto!(MadNLP.primal_dual(w), x)
    # Recover descents w.r.t. the bounds.
    MadNLP.finish_aug_solve!(kkt, w)
    return
end

function MadNLP.set_aug_RR!(kkt::BieglerKKTSystem, ips::MadNLP.MadNLPSolver, RR::MadNLP.RobustRestorer)
    x = MadNLP.full(ips.x)
    xl = MadNLP.full(ips.xl)
    xu = MadNLP.full(ips.xu)
    zl = MadNLP.full(ips.zl)
    zu = MadNLP.full(ips.zu)
    copyto!(kkt.pr_diag, zl./(x.-xl) .+ zu./(xu.-x) .+ RR.zeta.*RR.D_R.^2)
    copyto!(kkt.du_diag, .-RR.pp./RR.zp .- RR.nn./RR.zn)
end

