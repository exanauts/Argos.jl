struct ExaNLPModel{VT} <: NLPModels.AbstractNLPModel{Float64,Vector{Float64}}
    meta::NLPModels.NLPModelMeta{Float64, Vector{Float64}}
    counters::NLPModels.Counters
    nlp::AbstractNLPEvaluator
    hash_x::Vector{UInt64}
    # Sparsity pattern
    hrows::Vector{Int}
    hcols::Vector{Int}
    jrows::Vector{Int}
    jcols::Vector{Int}
    # CPU/GPU buffers
    d_x::VT
    d_g::VT
    d_c::VT
    # other buffers
    etc::Dict{Symbol, Any}
end
function ExaNLPModel(nlp::AbstractNLPEvaluator)
    n = n_variables(nlp)
    m = n_constraints(nlp)

    # Initial variable
    d_x0  = ExaOpt.initial(nlp)
    VT = typeof(d_x0)
    x0 = d_x0 |> Array
    y0 = zeros(m)
    # Bounds
    xl, xu = ExaOpt.bounds(nlp, ExaOpt.Variables()) .|> Array
    gl, gu = ExaOpt.bounds(nlp, ExaOpt.Constraints()) .|> Array
    # Buffers
    d_x = similar(d_x0, n)
    d_g = similar(d_x0, n)
    d_c = similar(d_x0, m)
    # Sparsity
    hrows = Int[]
    hcols = Int[]
    nnzh = div(n * (n + 1), 2)
    jrows = Int[]
    jcols = Int[]
    nnzj = n * m

    etc = Dict{Symbol, Any}()

    return ExaNLPModel{VT}(
        NLPModels.NLPModelMeta(
            n,
            ncon = m,
            nnzj = nnzj,
            nnzh = nnzh,
            x0 = x0,
            y0 = y0,
            lvar = xl,
            uvar = xu,
            lcon = gl,
            ucon = gu,
            minimize = true
        ),
        NLPModels.Counters(),
        nlp, UInt64[0],
        hrows, hcols, jrows, jcols,
        d_x, d_g, d_c, etc,
    )
end

function _update!(m::ExaNLPModel, x::AbstractVector)
    hx = hash(x)
    if hx != m.hash_x[1]
        xp = parent(x)
        n = length(m.d_x)
        copyto!(m.d_x, 1, xp, 1, n)
        update!(m.nlp, m.d_x)
        m.hash_x[1] = hx
    end
end

# Objective
function NLPModels.obj(m::ExaNLPModel,x)
    _update!(m, x)
    return objective(m.nlp, m.d_x)
end

# Gradient
function NLPModels.grad!(m::ExaNLPModel,x,g)
    _update!(m, x)
    gradient!(m.nlp, m.d_g, m.d_x)
    gp = parent(g)
    n = NLPModels.get_nvar(m)
    copyto!(gp, 1, m.d_g, 1, n)
    return
end

# Constraints
function NLPModels.cons!(m::ExaNLPModel,x,c)
    _update!(m, x)
    constraint!(m.nlp, m.d_c, m.d_x)
    cp = parent(c)
    _m = NLPModels.get_ncon(m)
    copyto!(cp, 1, m.d_c, 1, _m)
    return
end

# Jacobian: sparse callback
function NLPModels.jac_coord!(m::ExaNLPModel, x, jac::AbstractArray)
    _update!(m, x)
    n, m = n_variables(m.nlp), n_constraints(m.nlp)
    fill!(jac, 0)
    jac_v = view(jac, 1:(n*m))
    J = reshape(jac_v, m, n)
    jacobian!(m.nlp, J, x)
end

# Jacobian: dense callback
function MadNLP.jac_dense!(m::ExaNLPModel, x, J::AbstractMatrix)
    _update!(m, x)
    jacobian!(m.nlp, J, m.d_x)
end

# Hessian: sparse callback (work only on CPU)
function NLPModels.hess_coord!(m::ExaNLPModel,x, l, hess::AbstractVector; obj_weight=1.0)
    @assert isa(m.d_x, Array)
    n = n_variables(m.nlp)
    if !haskey(m.etc,:hess)
        m.etc[:hess] = zeros(n, n)
    end
    _update!(m, x)
    # Evaluate full reduced Hessian in the preallocated buffer.
    H = m.etc[:hess]
    hessian_lagrangian!(m.nlp, H, x, μ, σ)
    # Only dense Hessian supported now
    index = 1
    @inbounds for i in 1:n, j in 1:i
        # We average the values from the lower and upper triangles for stability.
        hess[index] = 0.5 * (H[i, j] + H[j, i])
        index += 1
    end
end

# Hessian: dense callback
function MadNLP.hess_dense!(m::ExaNLPModel, x, l, hess::AbstractMatrix; obj_weight=1.0)
    _update!(m, x)
    # Evaluate full reduced Hessian in the preallocated buffer.
    hessian_lagrangian!(m.nlp, hess, m.d_x, l, obj_weight)
end

# Special wrapper for Auglag

#=
    MixedAuglagKKTSystem
=#

# Supports only bound-constrained optimization problem (so no Jacobian)!
struct MixedAuglagKKTSystem{T, VT, MT} <: MadNLP.AbstractKKTSystem{T, MT}
    aug::AbstractNLPEvaluator # for Auglag information
    n::Int
    m::Int
    aug_com::MT
    hess::MT
    jac::MT
    jac_scaled::MT
    # Diagonal terms
    pr_diag::VT
    du_diag::VT
    sl_diag::VT
    diag_hess::VT
    # Buffers
    _wc::VT
    _wx::VT
    _wy::VT
    rhs::VT
    weights::VT
    # Scaling of IPM algorithm (mutable)
    ipp_scale::Vector{T}
    # Info
    ind_fixed::Vector{Int}
end

function MixedAuglagKKTSystem{T, VT, MT}(m::ExaNLPModel, ind_cons) where {T, VT, MT}

    aug = m.nlp
    inner = inner_evaluator(aug)
    n = n_variables(inner)
    m = n_constraints(inner)

    aug_com    = MT(undef, n, n)
    hess       = MT(undef, n, n)
    jac        = MT(undef, m, n)
    jac_scaled = MT(undef, m, n)
    diag_hess = VT(undef, n)
    pr_diag   = VT(undef, n + m) # Σ  = [Σᵤ, Σₛ]
    sl_diag   = VT(undef, m)     # Σₛ + ρ I
    rhs       = VT(undef, n + m)
    du_diag   = VT(undef, 0)
    weights   = VT(undef, m)
    _wc        = VT(undef, m)
    _wx        = VT(undef, n + m)
    _wy        = VT(undef, n + m)
    ipp_scale   = zeros(T, 1)

    # Init!
    fill!(aug_com,   zero(T))
    fill!(hess,      zero(T))
    fill!(jac,       zero(T))
    fill!(pr_diag,   zero(T))
    fill!(sl_diag,   zero(T))
    fill!(rhs    ,   zero(T))
    fill!(weights,   zero(T))

    return MixedAuglagKKTSystem{T, VT, MT}(
        aug, n, m, aug_com, hess, jac, jac_scaled,
        pr_diag, du_diag, sl_diag, diag_hess,
        _wc, _wx, _wy, rhs, weights, ipp_scale, ind_cons.ind_fixed,
    )
end

MadNLP.is_reduced(::MixedAuglagKKTSystem) = true

MadNLP.nnz_jacobian(kkt::MixedAuglagKKTSystem) = 0
MadNLP.nnz_kkt(kkt::MixedAuglagKKTSystem) = length(kkt.hess)
# We factorize only the Hessian part!
MadNLP.get_kkt(kkt::MixedAuglagKKTSystem) = kkt.aug_com
MadNLP.get_jacobian(kkt::MixedAuglagKKTSystem) = Float64[]

function MadNLP.build_kkt!(kkt::MixedAuglagKKTSystem{T, VT, MT}) where {T, VT, MT}
    ρ = kkt.aug.ρ  # current penalty
    σ  = kkt.ipp_scale[]
    ηcons = kkt.aug.scaler.scale_cons
    n = size(kkt.hess, 1)
    m = size(kkt.jac, 1)

    Σₛ = view(kkt.pr_diag, n+1:n+m)

    # Update sl_diag
    kkt.sl_diag .= Σₛ .+ ρ .* ηcons.^2 .* σ
    # Regularization
    ρₛ = view(kkt.rhs, n+1:m+n)
    ρₛ .= ρ .* (1.0 ./ (ηcons.^2 .* σ) .- ρ ./ kkt.sl_diag)

    # Huu
    copyto!(kkt.aug_com, kkt.hess)
    # Huu + Σᵤ (use MadNLP's function directly for GPU's support)
    MadNLP.diag_add!(kkt.aug_com, kkt.diag_hess, kkt.pr_diag)
    # Huu + Σᵤ + Aᵤ' * ρₛ * Aᵤ
    mul!(kkt.jac_scaled, Diagonal(ρₛ), kkt.jac)
    mul!(kkt.aug_com, kkt.jac', kkt.jac_scaled, 1.0, 1.0)

    return
end

function MadNLP.compress_hessian!(kkt::MixedAuglagKKTSystem)
    MadNLP.diag!(kkt.diag_hess, kkt.hess)
end

MadNLP.compress_jacobian!(kkt::MixedAuglagKKTSystem) = nothing
MadNLP.jtprod!(y::AbstractVector, kkt::MixedAuglagKKTSystem, x::AbstractVector) = nothing
MadNLP.set_jacobian_scaling!(kkt::MixedAuglagKKTSystem, constraint_scaling::AbstractVector) = nothing

function MadNLP.set_aug_diagonal!(kkt::MixedAuglagKKTSystem, ips::MadNLP.InteriorPointSolver)
    copyto!(kkt.pr_diag, ips.zl./(ips.x.-ips.xl) .+ ips.zu./(ips.xu.-ips.x))
    fill!(kkt.du_diag, 0.0)
end

function MadNLP.mul!(y::AbstractVector, kkt::MixedAuglagKKTSystem, x::AbstractVector)
    # Load problem
    n = kkt.n
    m = kkt.m
    σ  = kkt.ipp_scale[]
    ηcons = kkt.aug.scaler.scale_cons
    ρ = kkt.aug.ρ
    Σᵤ = view(kkt.pr_diag, 1:n)
    j = kkt._wc
    ρₛ = view(kkt.rhs, n+1:m+n)
    ρₛ .= ρ .* (1.0 ./ (ηcons.^2 .* σ))

    # Transfer on the device
    copyto!(kkt._wx, x)

    x_u = @view kkt._wx[1:n]
    x_s = @view kkt._wx[1+n:n+m]

    y_u = @view kkt._wy[1:n]
    y_s = @view kkt._wy[1+n:n+m]

    # Structure of the Hessian
    #= [
       H + Σᵤ + ρ * J' * D² * J        - ρ J' * D²  ;
           - ρ J' * D²                   ρ + Σₛ
       ]
    =#

    # Block (1, 1)
    mul!(y_u, kkt.hess, x_u)
    y_u .+= Σᵤ .* x_u
    mul!(j, kkt.jac, x_u)
    j .*= ρₛ
    mul!(y_u, kkt.jac', j, 1.0, 1.0)

    # Block (1, 2)
    mul!(y_u, kkt.jac', x_s, -ρ, 1.0)

    # Block (2, 2)
    y_s .= kkt.sl_diag .* x_s
    # Block (2, 1)
    mul!(y_s, kkt.jac, x_u, -ρ, 1.0)

    # Transfer back on the host
    copyto!(y, kkt._wy)
    return
end

# Overload Hessian evaluation
function MadNLP.eval_lag_hess_wrapper!(ipp::MadNLP.InteriorPointSolver, kkt::MixedAuglagKKTSystem, x::Vector{Float64},l::Vector{Float64};is_resto=false)
    nlp = ipp.nlp
    cnt = ipp.cnt
    # Scaling

    ηcons = kkt.aug.scaler.scale_cons
    λ = kkt._wc # avoid a new allocation
    λ .= kkt.aug.λc .* ηcons
    kkt.ipp_scale[] = ipp.obj_scale[]

    # Transfer on device
    copyto!(kkt._wx, x)

    n = kkt.n
    m = kkt.m
    D = kkt.weights
    xᵤ = @view kkt._wx[1:n]

    # Update Hessian-Lagrangian
    cnt.eval_function_time += @elapsed inner_hessian!(kkt.aug, kkt.hess, xᵤ, λ, D)
    kkt.hess .*= ipp.obj_scale[]

    # Update inner constraints' Jacobian
    cnt.eval_function_time += @elapsed inner_jacobian!(kkt.aug, kkt.jac, xᵤ)

    # Auglag's scaling D² * J
    kkt._wc .= ηcons.^2 .* ipp.obj_scale[] # avoid a new allocation
    mul!(kkt.jac, Diagonal(kkt._wc), kkt.jac)

    MadNLP.compress_hessian!(kkt)
    cnt.lag_hess_cnt+=1

    return kkt.hess
end

# Overload linear solve by Schur complement approach
function MadNLP.solve_refine_wrapper!(ipp::MadNLP.InteriorPointSolver{<:MixedAuglagKKTSystem}, x, b)
    kkt = ipp.kkt
    cnt = ipp.cnt
    ρ = kkt.aug.ρ # current penalty
    J = kkt.jac
    σ = kkt.ipp_scale[]

    n = kkt.n
    m = kkt.m

    MadNLP.fixed_variable_treatment_vec!(b, ipp.ind_fixed)

    copyto!(kkt._wy, b)

    x_u = @view kkt._wx[1:n]
    x_s = @view kkt._wx[1+n:n+m]

    b_u = @view kkt._wy[1:n]
    b_s = @view kkt._wy[1+n:n+m]

    rhs_u = @view kkt.rhs[1:n]
    rhs_s = @view kkt.rhs[1+n:n+m]

    # Init x_u
    copyto!(x_u, b_u)
    copyto!(rhs_s, b_s)
    rhs_s ./= kkt.sl_diag
    mul!(x_u, J', rhs_s, ρ, 1.0)

    cnt.linear_solver_time += @elapsed (MadNLP.solve!(ipp.linear_solver, x_u))

    copyto!(x_s, b_s)
    mul!(x_s, J, x_u, ρ, 1.0)
    x_s ./= kkt.sl_diag

    # Move back to the host
    copyto!(x, kkt._wx)

    MadNLP.fixed_variable_treatment_vec!(x, ipp.ind_fixed)
    return true
end

