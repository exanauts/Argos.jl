
struct BieglerKKTSystem{T, VI, VT, MT, SMT} <: MadNLP.AbstractReducedKKTSystem{T, MT}
    K::HJDJ{VI,VT,SMT}
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
    pr_diag::VT
    du_diag::VT
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
    # Dimension
    nx::Int
    nu::Int
    # Info
    ind_ineq::Vector{Int}
    ind_fixed::Vector{Int}
    con_scale::VT
    jacobian_scaling::VT
    etc::Dict{Symbol,Any}
end

function BieglerKKTSystem{T, VI, VT, MT}(nlp::ExaNLPModel, ind_cons=MadNLP.get_index_constraints(nlp); max_batches=256) where {T, VI, VT, MT}
    n_slack = length(ind_cons.ind_ineq)
    n = NLPModels.get_nvar(nlp)
    m = NLPModels.get_ncon(nlp)
    # Structure
    evaluator = backend(nlp)
    nx = evaluator.nx
    nu = evaluator.nu
    # Evaluate sparsity pattern
    nnzj = NLPModels.get_nnzj(nlp)
    nnzh = NLPModels.get_nnzh(nlp)

    W = evaluator.hess.H
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

    nbatches = min(max_batches, nu)
    linear_solver = LS.DirectSolver(Gx; nbatch=nbatches)
    Gxi = linear_solver.factorization
    S = ImplicitSensitivity(Gxi, Gu)
    reduction = if nbatches > 1
        BatchReduction(evaluator.model, S, nbatches)
    else
        Reduction(evaluator.model, S)
    end

    # W
    aug_com = MT(undef, nu, nu) ; fill!(aug_com, zero(T))

    # Buffers
    _wxu1 = VT(undef, nx+nu)
    _wxu2 = VT(undef, nx+nu)
    _wxu3 = VT(undef, nx+nu)
    _wx1 = VT(undef, nx)
    _wx2 = VT(undef, nx)
    _wj1 = VT(undef, m-nx)

    # Scaling
    con_scale = VT(undef, m)           ; fill!(con_scale, one(T))
    jacobian_scaling = VT(undef, nnzj) ; fill!(jacobian_scaling, one(T))

    ind_fixed = ind_cons.ind_fixed .- nx

    etc = Dict{Symbol, Any}(:reduction_time=>0.0)

    return BieglerKKTSystem{T, VI, VT, MT, SMT}(
        K, W, J, A, Gx, Gu, mapA, mapGx, mapGu,
        h_V, j_V,
        pr_diag, du_diag,
        aug_com, reduction, Gxi,
        _wxu1, _wxu2, _wxu3, _wx1, _wx2, _wj1,
        nx, nu,
        ind_cons.ind_ineq, ind_fixed, con_scale, jacobian_scaling,
        etc,
    )
end

MadNLP.num_variables(kkt::BieglerKKTSystem) = kkt.nu
MadNLP.get_hessian(kkt::BieglerKKTSystem) = kkt.h_V
MadNLP.get_jacobian(kkt::BieglerKKTSystem) = nonzeros(kkt.J)
MadNLP.is_reduced(::BieglerKKTSystem) = true

# Return SparseMatrixCOO to MadNLP
function MadNLP.get_raw_jacobian(kkt::BieglerKKTSystem)
    n, m = kkt.nx + kkt.nu, size(kkt.A, 1)
    i, j, v = findnz(kkt.J)
    return MadNLP.SparseMatrixCOO(n, m, i, j, v)
end

function MadNLP.initialize!(kkt::BieglerKKTSystem)
    fill!(kkt.pr_diag, 1.0)
    fill!(kkt.du_diag, 0.0)
end

function MadNLP.set_jacobian_scaling!(kkt::BieglerKKTSystem{T,VI,VT,MT}, constraint_scaling::AbstractVector) where {T,VI,VT,MT}
    copyto!(kkt.con_scale, constraint_scaling)
    nnzJ = length(kkt.j_V)::Int
    Ji, _, _ = findnz(kkt.J)
    jscale = zeros(nnzJ)
    @inbounds for i in 1:nnzJ
        index = Ji[i]
        jscale[i] = constraint_scaling[index]
    end
    copyto!(kkt.jacobian_scaling, jscale)
end

function _load_buffer(kkt::BieglerKKTSystem{T,VI,VT,MT}, x::AbstractVector, key::Symbol) where {T,VI,VT,MT}
    haskey(kkt.etc, key) || (kkt.etc[key] = VT(undef, length(x)))
    xb = kkt.etc[key]::VT
    _copyto!(xb, 1, x, 1, length(x))
    return xb
end
function _load_buffer(kkt::BieglerKKTSystem{T,VI,VT,MT}, x::VT, key::Symbol) where {T,VI,VT,MT}
    return x
end

# Use for inertia-free regularization (require full-space multiplication)
function _mul_expanded!(y_h, kkt::BieglerKKTSystem{T, VI, VT, MT}, x_h) where {T, VI, VT, MT}
    x = _load_buffer(kkt, x_h, :hess_x)::VT
    y = _load_buffer(kkt, y_h, :hess_y)::VT
    # Build full-space KKT system
    n = kkt.nx + kkt.nu
    m = length(kkt.con_scale)
    ns = length(kkt.ind_ineq)
    W = kkt.W
    Σₓ = kkt.pr_diag[1:n]
    Σₛ = kkt.pr_diag[n+1:n+ns]
    # Jacobian
    Jx = kkt.J
    Js = view(kkt.con_scale, kkt.nx+1:m)
    Σᵤ = kkt.du_diag

    # Decompose x
    xx = view(x, 1:n)
    xs = view(x, n+1:n+ns)
    xz = view(x, n+ns+1:n+ns+m)
    # Decompose y
    yx = view(y, 1:n)
    ys = view(y, n+1:n+ns)
    yz = view(y, n+ns+1:n+ns+m)

    # / x (variable)
    yx .= Σₓ .* xx
    mul!(yx, W, xx, 1.0, 1.0)
    mul!(yx, Jx', xz, 1.0, 1.0)
    # / s (slack)
    ys .= Σₛ .* xs
    ys .-= Js .* xz[kkt.nx+1:m]
    # / z (multipliers)
    yz .= Σᵤ .* xz
    yz[kkt.nx+1:m] .-= Js .* xs
    mul!(yz, Jx, xx, 1.0, 1.0)

    copyto!(y_h, y)
end

function MadNLP.mul!(y::AbstractVector, kkt::BieglerKKTSystem, x::AbstractVector)
    if size(kkt.aug_com, 1) == length(x) == length(y)
        mul!(y, kkt.aug_com, x)
    else
        _mul_expanded!(y, kkt, x)
    end
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
    αs = view(kkt.con_scale, kkt.nx+1:m)
    ys .= .-xs .* αs

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
    return
end

# Build reduced Hessian
MadNLP.compress_hessian!(kkt::BieglerKKTSystem) = nothing

function assemble_condensed_matrix!(kkt::BieglerKKTSystem, K::HJDJ)
    nx, nu = kkt.nx, kkt.nu
    m = size(kkt.J, 1)
    ns = size(kkt.A, 1)
    D = kkt._wj1
    α   = @view kkt.con_scale[nx+1:m]
    prx = @view kkt.pr_diag[1:nx+nu]
    prs = @view kkt.pr_diag[nx+nu+1:nx+nu+ns]
    # Matrices
    A = kkt.A
    D .= prs ./ α.^2
    update!(K, A, D, prx)
end

function MadNLP.build_kkt!(kkt::BieglerKKTSystem{T, VI, VT, MT}) where {T, VI, VT, MT}
    assemble_condensed_matrix!(kkt, kkt.K)
    fill!(kkt.aug_com, 0.0)
    update!(kkt.reduction)
    kkt.etc[:reduction_time] += @elapsed begin
        reduce!(kkt.reduction, kkt.aug_com, kkt.K)
    end
    MadNLP.treat_fixed_variable!(kkt)
end

function MadNLP.solve_refine_wrapper!(
    ips::MadNLP.InteriorPointSolver{<:BieglerKKTSystem{T,VI,VT,MT}},
    x_h, b_h,
) where {T, VI, VT, MT}
    kkt = ips.kkt
    x = _load_buffer(kkt, x_h, :kkt_x)::VT
    b = _load_buffer(kkt, b_h, :kkt_b)::VT
    MadNLP.fixed_variable_treatment_vec!(b, ips.ind_fixed)
    m = ips.m # constraints
    nx, nu = kkt.nx, kkt.nu
    ns = m - nx
    @assert length(b) == length(x) == ips.n + m


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
    α = view(kkt.con_scale, nx+1:m)

    # RHS
    r₁₂ = view(b, 1:nx+nu)
    r₁ = view(b, 1:nx)                   # / state
    r₂ = view(b, 1+nx:nx+nu)             # / control
    r₃ = view(b, 1+nx+nu:ips.n)          # / slack
    r₄ = view(b, ips.n+1:ips.n+nx)       # / equality cons
    r₅ = view(b, ips.n+nx+1:ips.n+m)     # / inequality cons
    # LHS
    dxu = view(x, 1:nx+nu)
    dx = view(x, 1:nx)                   # / state
    du = view(x, 1+nx:nx+nu)             # / control
    ds = view(x, 1+nx+nu:ips.n)          # / slack
    dλ = view(x, ips.n+1:ips.n+nx)       # / equality cons
    dy = view(x, ips.n+nx+1:ips.n+m)     # / inequality cons


    # Reduction (1) --- Condensed
    vj .= (Σₛ .* r₅ .+ α .* r₃) ./ α.^2   # v = (α Σₛ⁻¹ α)⁻¹ * (r₅ + α Σₛ⁻¹ r₃)
    mul!(jv, kkt.A', vj)                  # jᵥ = Aᵀ v
    jv .+= r₁₂                            # r₁₂ - Aᵀv
    # Reduction (2) --- Biegler
    sx1 .= r₄                             # r₄
    ldiv!(Gxi, sx1)                       # Gₓ⁻¹ r₄
    sx2 .= tx                             # tx = jv[1:nx]
    kvx .= sx1 ; kvu .= 0.0
    mul!(kh, K, kv)                       # [Kₓₓ Gₓ⁻¹ r₄ ; Kᵤₓ Gₓ⁻¹ r₄ ]
    sx2 .= khx .- tx                      # sₓ = Kₓₓ Gₓ⁻¹ r₄ - tₓ
    ldiv!(Gxi', sx2)                      # Gₓ⁻ᵀ sₓ
    mul!(tu, Gu', sx2, 1.0, 1.0)          # tᵤ = tᵤ + Gᵤᵀ Gₓ⁻ᵀ sₓ
    axpy!(-1.0, khu, tu)                  # tᵤ = tᵤ - Kᵤₓ Gₓ⁻¹ r₄

    du .= tu
    ips.cnt.linear_solver_time += @elapsed begin
        MadNLP.solve!(ips.linear_solver, du)
    end
    solve_status = true

    # (1) Extract Biegler
    dx .= r₄                              # r₄
    mul!(dx, Gu, du, -1.0, 1.0)           # r₄ - Gᵤ dᵤ
    ldiv!(Gxi, dx)                        # dₓ = Gₓ⁻¹ (r₄ - Gᵤ dᵤ)
    dλ .= tx                              # tₓ
    mul!(kh, K, dxu)                      # Kₓₓ dₓ + Kₓᵤ dᵤ
    axpy!(-1.0, khx, dλ)                  # tₓ - Kₓₓ dₓ + Kₓᵤ dᵤ
    ldiv!(Gxi', dλ)                       # dₗ = Gₓ⁻ᵀ(tₓ - Kₓₓ dₓ + Kₓᵤ dᵤ)
    # (2) Extract Condensed
    mul!(vj, kkt.A, dxu)                  # Aₓ dₓ + Aᵤ dᵤ
    dy .= Σₛ .* (vj .- r₅) ./ α.^2 .- r₃ ./ α
    ds .= (r₃ .+ α .* dy) ./ Σₛ

    MadNLP.fixed_variable_treatment_vec!(x, ips.ind_fixed)
    copyto!(x_h, x)
    return solve_status
end

