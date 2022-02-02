struct ExaNLPModel{VT,Evaluator} <: NLPModels.AbstractNLPModel{Float64,Vector{Float64}}
    meta::NLPModels.NLPModelMeta{Float64, Vector{Float64}}
    counters::NLPModels.Counters
    timers::NLPTimers
    nlp::Evaluator
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
end
function ExaNLPModel(nlp::AbstractNLPEvaluator)
    n = n_variables(nlp)
    m = n_constraints(nlp)

    # Initial variable
    d_x0  = initial(nlp)
    VT = typeof(d_x0)
    x0 = d_x0 |> Array
    y0 = zeros(m)
    # Bounds
    xl, xu = bounds(nlp, Variables()) .|> Array
    gl, gu = bounds(nlp, Constraints()) .|> Array
    # Buffers
    d_x = similar(d_x0, n)
    d_g = similar(d_x0, n)
    d_c = similar(d_x0, m)
    # Sparsity
    hrows, hcols = hessian_structure(nlp)
    jrows, jcols = jacobian_structure(nlp)
    nnzh = length(hrows)
    nnzj = length(jrows)

    return ExaNLPModel{VT, typeof(nlp)}(
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
        NLPTimers(),
        nlp, UInt64[0],
        hrows, hcols, jrows, jcols,
        d_x, d_g, d_c,
    )
end

function NLPModels.jac_structure!(m::ExaNLPModel, rows, cols)
    copyto!(rows, m.jrows)
    copyto!(cols, m.jcols)
    return rows, cols
end

function NLPModels.hess_structure!(m::ExaNLPModel, rows, cols)
    copyto!(rows, m.hrows)
    copyto!(cols, m.hcols)
    return rows, cols
end

function _update!(m::ExaNLPModel, x::AbstractVector)
    hx = hash(x)
    if hx != m.hash_x[1]
        xp = parent(x)
        n = length(m.d_x)
        copyto!(m.d_x, 1, xp, 1, n)
        m.timers.update_time += @elapsed begin
            update!(m.nlp, m.d_x)
        end
        m.hash_x[1] = hx
    end
end

# Objective
function NLPModels.obj(m::ExaNLPModel,x)
    _update!(m, x)
    m.timers.obj_time += @elapsed begin
        obj = objective(m.nlp, m.d_x)
    end
    return obj
end

# Gradient
function NLPModels.grad!(m::ExaNLPModel,x,g)
    _update!(m, x)
    m.timers.grad_time += @elapsed begin
        gradient!(m.nlp, m.d_g, m.d_x)
    end
    gp = parent(g)
    n = NLPModels.get_nvar(m)
    copyto!(gp, 1, m.d_g, 1, n)
    return
end

# Constraints
function NLPModels.cons!(m::ExaNLPModel,x,c)
    _update!(m, x)
    m.timers.cons_time += @elapsed begin
        constraint!(m.nlp, m.d_c, m.d_x)
    end
    cp = parent(c)
    _m = NLPModels.get_ncon(m)
    copyto!(cp, 1, m.d_c, 1, _m)
    return
end

# Jacobian: sparse callback
function NLPModels.jac_coord!(m::ExaNLPModel, x, jac::AbstractArray)
    _update!(m, x)
    nnzj = NLPModels.get_nnzj(m)
    jv = view(jac, 1:nnzj)
    m.timers.jacobian_time += @elapsed begin
        jacobian_coo!(m.nlp, jv, m.d_x)
    end
end

# Jacobian: dense callback
function MadNLP.jac_dense!(m::ExaNLPModel, x, J::AbstractMatrix)
    _update!(m, x)
    m.timers.jacobian_time += @elapsed begin
        jacobian!(m.nlp, J, m.d_x)
    end
end


# Hessian-vector products
function NLPModels.hprod!(m::ExaNLPModel, x, l, v, hv::AbstractVector; obj_weight=1.0)
    m.timers.hessprod_time += @elapsed begin
        hessian_lagrangian_prod!(m.nlp, hv, x, l, obj_weight, v)
    end
end

function _copyto!(dest::AbstractArray, off1, src::Array, off2, n)
    copyto!(dest, off1, src, off2, n)
end
function _copyto!(dest::AbstractArray, off1, src::SubArray, off2, n)
    p_src = parent(src)
    copyto!(dest, off1, p_src, off2 + src.offset1, n)
end

# Hessian: sparse callback
function NLPModels.hess_coord!(m::ExaNLPModel,x, l, hess::AbstractVector; obj_weight=1.0)
    _copyto!(m.d_c, 1, l, 1, NLPModels.get_ncon(m))
    m.timers.hessian_time += @elapsed begin
        hessian_lagrangian_coo!(m.nlp, hess, x, m.d_c, obj_weight)
    end
end

# Hessian: dense callback
function MadNLP.hess_dense!(m::ExaNLPModel, x, l, hess::AbstractMatrix; obj_weight=1.0)
    _update!(m, x)
    _copyto!(m.d_c, 1, l, 1, NLPModels.get_ncon(m))
    # Evaluate full reduced Hessian in the preallocated buffer.
    m.timers.hessian_time += @elapsed begin
        hessian_lagrangian!(m.nlp, hess, m.d_x, m.d_c, obj_weight)
    end
end
