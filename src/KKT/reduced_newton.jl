
struct BieglerKKTSystem{T, VI, VT, MT} <: MadNLP.AbstractReducedKKTSystem{T, MT}
    # Sparse Hessian in COO format
    h_V::VT
    h_I::VI
    h_J::VI
    # Sparse Jacobian in COO format
    j_V::VT
    j_I::VI
    j_J::VI
    # Regularization terms
    pr_diag::VT
    du_diag::VT
    # Reduced KKT system
    aug_com::MT
    # Raw Hessian
    hess_raw::MT
    # Jacobian
    jac_raw::MT
    # Sensitivities
    S::MT
    # Pivot
    G_fac::LinearAlgebra.Factorization
    # Dimension
    nx::Int
    nu::Int
    # Info
    ind_ineq::Vector{Int}
    ind_fixed::Vector{Int}
    con_scale::VT
    jacobian_scaling::VT
end

function BieglerKKTSystem{T, VI, VT, MT}(nlp::ExaNLPModel, ind_cons=MadNLP.get_index_constraints(nlp)) where {T, VI, VT, MT}
    n_slack = length(ind_cons.ind_ineq)
    n = NLPModels.get_nvar(nlp)
    m = NLPModels.get_ncon(nlp)
    # Evaluate sparsity pattern
    nnzj = NLPModels.get_nnzj(nlp)
    nnzh = NLPModels.get_nnzh(nlp)

    h_V = VT(undef, nnzh) ; fill!(h_V, zero(T))
    h_I = VT(undef, nnzh)
    h_J = VT(undef, nnzh)
    NLPModels.hess_structure!(nlp, h_I, h_J)

    j_V = VT(undef, nnzj) ; fill!(j_V, zero(T))
    j_I = VT(undef, nnzj)
    j_J = VT(undef, nnzj)
    NLPModels.jac_structure!(nlp, j_I, j_J)

    pr_diag = VT(undef, n + n_slack) ; fill!(pr_diag, zero(T))
    du_diag = VT(undef, m) ; fill!(du_diag, zero(T))

    # Reduction
    nx = nlp.nlp.nx
    nu = nlp.nlp.nu

    n_kkt = nu + n_slack + m - nx

    # Sensitivities
    S = MT(undef, nx, nu) ; fill!(S, 0.0)
    # Initial factorization
    # Evaluate Jacobian
    x = initial(nlp.nlp)
    NLPModels.jac_coord!(nlp, x, j_V)
    J = sparse(j_I, j_J, j_V)
    Gx = J[1:nx, 1:nx]
    Gxi = lu(Gx)

    # Wᵤᵤ
    hess_raw = MT(undef, nu, nu)      ; fill!(hess_raw, zero(T))
    # Jᵤ
    jac_raw = MT(undef, m - nx, nu)   ; fill!(jac_raw, zero(T))
    # W
    aug_com = MT(undef, n_kkt, n_kkt) ; fill!(aug_com, zero(T))

    # Scaling
    con_scale = VT(undef, m)           ; fill!(con_scale, one(T))
    jacobian_scaling = VT(undef, nnzj) ; fill!(jacobian_scaling, one(T))

    ind_fixed = ind_cons.ind_fixed .- nx

    return BieglerKKTSystem{T, VI, VT, MT}(
        h_V, h_I, h_J,
        j_V, j_I, j_J,
        pr_diag, du_diag,
        aug_com, hess_raw, jac_raw, S, Gxi,
        nx, nu,
        ind_cons.ind_ineq, ind_fixed, con_scale, jacobian_scaling,
    )
end

MadNLP.num_variables(kkt::BieglerKKTSystem) = kkt.nu + length(kkt.ind_ineq)
MadNLP.get_hessian(kkt::BieglerKKTSystem) = kkt.h_V
MadNLP.get_jacobian(kkt::BieglerKKTSystem) = kkt.j_V
MadNLP.is_reduced(::BieglerKKTSystem) = true

# Return SparseMatrixCOO to MadNLP
function MadNLP.get_raw_jacobian(kkt::BieglerKKTSystem)
    n, m = kkt.nx + kkt.nu, size(kkt.jac_raw, 1)
    return MadNLP.SparseMatrixCOO(n, m, kkt.j_I, kkt.j_J, kkt.j_V)
end

function MadNLP.initialize!(kkt::BieglerKKTSystem)
    fill!(kkt.pr_diag, 1.0)
    fill!(kkt.du_diag, 0.0)
    fill!(kkt.hess_raw, 0.0)
end

# Use for inertia-free regularization (require full-space multiplication)
# TODO: rewrite
function _mul_expanded!(y, kkt::BieglerKKTSystem, x)
    # Build full-space KKT system
    n = kkt.nx + kkt.nu
    m = length(kkt.con_scale)
    ns = length(kkt.ind_ineq)
    W = sparse(kkt.h_I, kkt.h_J, kkt.h_V)
    Σₓ = spdiagm(kkt.pr_diag[1:n])
    Σₛ = spdiagm(kkt.pr_diag[n+1:n+ns])
    # Jacobian
    Jx = sparse(kkt.j_I, kkt.j_J, kkt.j_V)
    Js = sparse(kkt.ind_ineq, 1:ns, -kkt.con_scale[kkt.ind_ineq], m, ns)
    Σᵤ = spdiagm(kkt.du_diag)

    Z = spzeros(n, ns)
    K = [
        W+Σₓ Z  Jx'
        Z'   Σₛ Js'
        Jx   Js Σᵤ
    ]::SparseMatrixCSC{Float64, Int}
    mul!(y, K, x)
end

function MadNLP.mul!(y::AbstractVector, kkt::BieglerKKTSystem, x::AbstractVector)
    if size(kkt.aug_com, 1) == length(x) == length(y)
        mul!(y, kkt.aug_com, x)
    else
        _mul_expanded!(y, kkt, x)
    end
end

function MadNLP.jtprod!(y::AbstractVector, kkt::BieglerKKTSystem, x::AbstractVector)
    n = kkt.nx + kkt.nu
    nv = length(y)
    J = sparse(kkt.j_I, kkt.j_J, kkt.j_V)
    yx = view(y, 1:n)
    mul!(yx, J', x)
    ys = view(y, n+1:nv)
    ys .= -x[kkt.ind_ineq] .* kkt.con_scale[kkt.ind_ineq]
end

MadNLP.nnz_jacobian(kkt::BieglerKKTSystem) = size(kkt.jac_raw, 1) * size(kkt.jac_raw, 2)

function MadNLP.compress_jacobian!(kkt::BieglerKKTSystem)
    nx, nu = kkt.nx, kkt.nu
    kkt.j_V .*= kkt.jacobian_scaling
    # Build Jacobian
    S = kkt.S
    J = sparse(kkt.j_I, kkt.j_J, kkt.j_V)
    Gx = J[1:nx, 1:nx]
    Gu = J[1:nx, 1+nx:nx+nu]
    Ax = J[nx+1:end, 1:nx]
    Au = J[nx+1:end, 1+nx:nx+nu]

    Gxi = kkt.G_fac
    lu!(Gxi, Gx)
    # update sensitivity matrix
    LinearAlgebra.ldiv!(kkt.S, Gxi, Gu)
    # Reduced Jacobian
    kkt.jac_raw .= Au - Ax * S
    return
end

# Build reduced Hessian
MadNLP.compress_hessian!(kkt::BieglerKKTSystem) = nothing

function _build_dense_kkt_system!(dest, hess, jac, pr_diag, du_diag, con_scale, ind_ineq, nu, nx, m, ns)
    # Transfer Hessian
    for i in 1:nu, j in 1:i
        if i == j
            dest[i, i] = hess[i, i]
        else
            dest[i, j] = hess[i, j]
            dest[j, i] = hess[j, i]
        end
    end
    # Transfer slack diagonal
    for i in 1:ns
        dest[i+nu, i+nu] = pr_diag[i+nu + nx]
    end
    # Transfer Jacobian
    for i in 1:m, j in 1:nu
        dest[i + nu + ns, j] = jac[i, j]
        dest[j, i + nu + ns] = jac[i, j]
    end
    # Transfer slack Jacobian
    for i in 1:ns
        is = ind_ineq[i] - nx
        dest[is + nu + ns, is + nu] = - con_scale[ind_ineq[i]]
        dest[is + nu, is + nu + ns] = - con_scale[ind_ineq[i]]
    end
    # Transfer dual regularization
    for i in 1:m
        dest[i + nu + ns, i + nu + ns] = du_diag[i]
    end
end

function MadNLP.build_kkt!(kkt::BieglerKKTSystem{T, VI, VT, MT}) where {T, VI, VT, MT}
    # Assemble reduced Hessian
    nx, nu = kkt.nx, kkt.nu
    S = kkt.S
    W = sparse(kkt.h_I, kkt.h_J, kkt.h_V)
    Wxx = Symmetric(W[1:nx, 1:nx], :L)
    Wux = W[1+nx:nx+nu, 1:nx]
    Wuu = Symmetric(W[1+nx:nx+nu, 1+nx:nx+nu], :L)

    Σₓ = spdiagm(kkt.pr_diag[1:nx])
    Σᵤ = spdiagm(kkt.pr_diag[1+nx:nx+nu])

    kkt.hess_raw .= Wuu + Σᵤ - S' * Wux' - Wux * S + S' * (Wxx + Σₓ) * S

    m = size(kkt.jac_raw, 1)
    ns = length(kkt.ind_ineq)
    # Assemble final matrix used in factorization
    fill!(kkt.aug_com, 0.0)
    _build_dense_kkt_system!(kkt.aug_com, kkt.hess_raw, kkt.jac_raw,
                             kkt.pr_diag, kkt.du_diag, kkt.con_scale, kkt.ind_ineq, nu, nx, m, ns)

    MadNLP.treat_fixed_variable!(kkt)
end

function MadNLP.set_jacobian_scaling!(kkt::BieglerKKTSystem, constraint_scaling::AbstractVector)
    copyto!(kkt.con_scale, constraint_scaling)
    nnzJ = length(kkt.j_V)::Int
    @inbounds for i in 1:nnzJ
        index = kkt.j_I[i]
        kkt.jacobian_scaling[i] = constraint_scaling[index]
    end
end

function MadNLP.solve_refine_wrapper!(ips::MadNLP.InteriorPointSolver{<:BieglerKKTSystem}, x, b)
    kkt = ips.kkt
    MadNLP.fixed_variable_treatment_vec!(b, ips.ind_fixed)
    m = ips.m # constraints
    ns = length(ips.ind_ineq) # slack
    nx, nu = kkt.nx, kkt.nu

    @assert length(b) == length(x) == ips.n + m

    dx = view(x, 1:nx)
    du = view(x, 1+nx:nx+nu)
    ds = view(x, 1+nx+nu:ips.n)
    dλ = view(x, ips.n+1:ips.n+nx)
    dy = view(x, ips.n+nx+1:ips.n+m)

    W = sparse(kkt.h_I, kkt.h_J, kkt.h_V)
    Wxx = Symmetric(W[1:nx, 1:nx], :L)
    Wux = W[1+nx:nx+nu, 1:nx]

    Gxi = kkt.G_fac
    S = kkt.S
    prₓ = view(kkt.pr_diag, 1:nx)
    Σₓ = spdiagm(prₓ)

    # Jacobian
    J = sparse(kkt.j_I, kkt.j_J, kkt.j_V)
    Gx = J[1:nx, 1:nx]
    Gu = J[1:nx, 1+nx:nx+nu]
    Ax = J[nx+1:m, 1:nx]
    Au = J[nx+1:m, 1+nx:nx+nu]

    r = -b
    # Build RHS
    r₁ = view(r, 1:nx)               # / state
    r₂ = view(r, 1+nx:nx+nu)         # / control
    r₃ = view(r, 1+nx+nu:ips.n)      # / slack
    r₄ = view(r, ips.n+1:ips.n+nx)   # / equality cons
    r₅ = view(r, ips.n+nx+1:ips.n+m) # / inequality cons

    # (du, ds, dy)
    gr = zeros(nx)
    ldiv!(gr, Gxi, r₄)
    xr = zeros(nu + ns + m - nx)
    br = zeros(nu + ns + m - nx)
    # Init RHS
    br[1:nu] .= -r₂ + S' * r₁ + Wux * gr - S' * (Wxx + Σₓ) * gr
    # Second order correction
    br[1+nu:nu+ns] .= -r₃            # / slack
    br[1+nu+ns:end] .= -r₅ + Ax * gr # / constraint

    # Linear solver
    ips.cnt.linear_solver_time += @elapsed (result = MadNLP.solve_refine!(xr, ips.iterator, br))
    solve_status = (result == :Solved)

    # Get solution
    du .= xr[1:nu]
    ds .= xr[1+nu:nu+ns]
    dy .= xr[1+nu+ns:end]

    # State descent direction
    rhs = -r₄ - Gu * du
    ldiv!(dx, Gxi, rhs)

    # Adjoint descent direction
    rhs2 = -(r₁ + Ax' * dy + Wux' * du + (Wxx + Σₓ) * dx)
    ldiv!(dλ, Gxi', rhs2)

    MadNLP.fixed_variable_treatment_vec!(x, ips.ind_fixed)
    return solve_status
end

