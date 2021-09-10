function MadNLP.NonlinearProgram(nlp::ExaOpt.AbstractNLPEvaluator)

    n = ExaOpt.n_variables(nlp)
    m = ExaOpt.n_constraints(nlp)

    hrows, hcols = ExaOpt.hessian_structure(nlp)
    jrows, jcols = ExaOpt.jacobian_structure(nlp)
    nnz_hess = length(hrows)
    nnz_jac = length(jrows)

    d_x0  = ExaOpt.initial(nlp)
    x0 = d_x0 |> Array
    g  = Vector{Float64}(undef,m)
    xl, xu = ExaOpt.bounds(nlp, ExaOpt.Variables()) .|> Array
    zl = Vector{Float64}(undef,n)
    zu = Vector{Float64}(undef,n)

    l = Vector{Float64}(undef,m)
    gl, gu = ExaOpt.bounds(nlp, ExaOpt.Constraints()) .|> Array

    d_x = similar(d_x0, n)
    d_g = similar(d_x0, n)
    d_c = similar(d_x0, m)

    hash_x = UInt64(0)

    function _update!(x::AbstractVector)
        hx = hash(x)
        if hx != hash_x
            hash_x = hx
            copyto!(d_x, x)
            ExaOpt.update!(nlp, d_x)
        end
    end
    function obj(x::AbstractArray{Float64,1})
        _update!(x)
        return ExaOpt.objective(nlp, d_x)
    end
    function obj_grad!(g::AbstractArray{Float64,1},x::AbstractArray{Float64,1})
        _update!(x)
        ExaOpt.gradient!(nlp, d_g, d_x)
        copyto!(g, d_g)
    end
    function con!(c::Array{Float64,1},x::AbstractArray{Float64,1})
        _update!(x)
        ExaOpt.constraint!(nlp, d_c, d_x)
        copyto!(c, d_c)
    end
    function con_jac!(jac::AbstractArray{Float64,1}, x::AbstractArray{Float64,1})
        # _update!(x)
        # J = reshape(jac, m, n)
        # ExaOpt.jacobian!(nlp, J, x)
    end
    function con_jac!(jac::AbstractArray{Float64,2}, x::AbstractArray{Float64,1})
        # _update!(x)
        # ExaOpt.jacobian!(nlp, jac, d_x)
    end
    function lag_hess!(hess::AbstractArray{Float64,2},x::AbstractArray{Float64,1},l::AbstractArray{Float64,1}, sig::Float64)
        _update!(x)
        # Evaluate full reduced Hessian in the preallocated buffer.
        ExaOpt.hessian!(nlp, hess, d_x)
        hess .*= sig
        return
    end
    function hess_sparsity!(I,J)
        copy!(I, hrows)
        copy!(J, hcols)
        return
    end
    function jac_sparsity!(I,J)
        copy!(I, jrows)
        copy!(J, jcols)
        return
    end

    # Build MadNLP model
    return MadNLP.NonlinearProgram(
        n, m, nnz_hess, nnz_jac, 0., x0, g, l, zl, zu, xl, xu, gl, gu,
        obj, obj_grad!, con!, con_jac!, lag_hess!, hess_sparsity!, jac_sparsity!,
        MadNLP.INITIAL,
        Dict{Symbol, Any}()
    )
end

# Special wrapper for Auglag

#=
    MixedAuglagKKTSystem
=#

# Supports only bound-constrained optimization problem (so no Jacobian)!
struct MixedAuglagKKTSystem{T, VT, MT} <: MadNLP.AbstractKKTSystem{T, MT}
    aug::AugLagEvaluator # for Auglag information
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

function MixedAuglagKKTSystem{T, VT, MT}(aug::AugLagEvaluator, ind_fixed) where {T, VT, MT}
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
    fill!(weights,   zero(T))

    return MixedAuglagKKTSystem{T, VT, MT}(
        aug, aug_com, hess, jac, jac_scaled,
        pr_diag, du_diag, sl_diag, diag_hess,
        _wc, _wx, _wy, rhs, weights, ipp_scale, ind_fixed,
    )
end

MadNLP.is_reduced(::MixedAuglagKKTSystem) = true

MadNLP.nnz_jacobian(kkt::MixedAuglagKKTSystem) = 0
MadNLP.nnz_kkt(kkt::MixedAuglagKKTSystem) = length(kkt.hess)
# We factorize only the Hessian part!
MadNLP.get_kkt(kkt::MixedAuglagKKTSystem) = kkt.aug_com

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

function MadNLP.mul!(y::AbstractVector, kkt::MixedAuglagKKTSystem, x::AbstractVector)
    # Load problem
    inner = inner_evaluator(kkt.aug)
    n = n_variables(inner)
    m = n_constraints(inner)
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
function MadNLP.eval_lag_hess_wrapper!(ipp::MadNLP.Solver, kkt::MixedAuglagKKTSystem, x::Vector{Float64},l::Vector{Float64};is_resto=false)
    kkt.aug.counter.hessian += 1
    nlp = ipp.nlp
    cnt = ipp.cnt
    # Scaling
    ηcons = kkt.aug.scaler.scale_cons
    σ = kkt.aug.scaler.scale_obj
    λ = kkt._wc # avoid a new allocation
    λ .= kkt.aug.λc .* ηcons
    kkt.ipp_scale[] = ipp.obj_scale[]

    # Transfer on device
    copyto!(kkt._wx, x)

    inner = inner_evaluator(kkt.aug)
    n = n_variables(inner)
    m = n_constraints(inner)
    D = kkt.weights
    xᵤ = @view kkt._wx[1:n]

    # Update Hessian-Lagrangian
    cnt.eval_function_time += @elapsed hessian_lagrangian_penalty!(inner, kkt.hess, xᵤ, λ, σ, D)
    kkt.hess .*= ipp.obj_scale[]

    # Update inner constraints' Jacobian
    cnt.eval_function_time += @elapsed jacobian!(inner, kkt.jac, xᵤ)
    # Auglag's scaling D² * J
    kkt._wc .= ηcons.^2 .* ipp.obj_scale[] # avoid a new allocation
    mul!(kkt.jac, Diagonal(kkt._wc), kkt.jac)

    MadNLP.compress_hessian!(kkt)
    cnt.lag_hess_cnt+=1

    if !isnothing(kkt.aug.tracker)
        store!(kkt.aug, kkt.aug.tracker, kkt._wx)
    end

    return kkt.hess
end

# Overload linear solve by Schur complement approach
function MadNLP.solve_refine_wrapper!(ipp::MadNLP.Solver{<:MixedAuglagKKTSystem}, x, b)
    kkt = ipp.kkt
    cnt = ipp.cnt
    ρ = kkt.aug.ρ # current penalty
    J = kkt.jac
    σ = kkt.ipp_scale[]

    inner = inner_evaluator(kkt.aug)
    n = n_variables(inner)
    m = n_constraints(inner)

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

