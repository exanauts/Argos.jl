struct ExaNLPModel{Evaluator} <: NLPModels.AbstractNLPModel{Float64,Vector{Float64}}
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
end

function ExaNLPModel(nlp::AbstractNLPEvaluator)
    n = n_variables(nlp)
    m = n_constraints(nlp)

    # Initial variable
    y0 = zeros(m)
    # Bounds
    xl, xu = bounds(nlp, Variables())
    gl, gu = bounds(nlp, Constraints())
    x0  = initial(nlp)
    # x0  = 0.5 .* (xl .+ xu)
    # x0[isinf.(x0)] .= 0.0
    # x0[isnan.(x0)] .= 0.0
    # Sparsity
    hrows, hcols = hessian_structure(nlp)
    jrows, jcols = jacobian_structure(nlp)
    nnzh = length(hrows)
    nnzj = length(jrows)

    return ExaNLPModel{typeof(nlp)}(
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
    )
end

backend(m::ExaNLPModel) = backend(m.nlp)

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
        m.timers.update_time += @elapsed begin
            update!(m.nlp, x)
        end
    end
end

# Objective
function NLPModels.obj(m::ExaNLPModel,x)
    _update!(m, x)
    m.timers.obj_time += @elapsed begin
        obj = objective(m.nlp, x)
    end
    return obj
end

# Gradient
function NLPModels.grad!(m::ExaNLPModel, x, g)
    _update!(m, x)
    m.timers.grad_time += @elapsed begin
        gradient!(m.nlp, g, x)
    end
    return
end

# Constraints
function NLPModels.cons!(m::ExaNLPModel,x,c)
    _update!(m, x)
    m.timers.cons_time += @elapsed begin
        constraint!(m.nlp, c, x)
    end
    return
end

# Jacobian: sparse callback
function NLPModels.jac_coord!(m::ExaNLPModel, x, jac::AbstractArray)
    _update!(m, x)
    nnzj = NLPModels.get_nnzj(m)
    jv = view(jac, 1:nnzj) # NB: ensure compatiliby with MadNLP
    m.timers.jacobian_time += @elapsed begin
        jacobian_coo!(m.nlp, jv, x)
    end
end

# Jacobian: dense callback
function MadNLP.jac_dense!(m::ExaNLPModel, x, J::AbstractMatrix)
    _update!(m, x)
    m.timers.jacobian_time += @elapsed begin
        jacobian!(m.nlp, J, x)
    end
end

# Hessian-vector products
function NLPModels.hprod!(m::ExaNLPModel, x, l, v, hv::AbstractVector; obj_weight=1.0)
    m.timers.hessprod_time += @elapsed begin
        hessian_lagrangian_prod!(m.nlp, hv, x, l, obj_weight, v)
    end
end

# Hessian: sparse callback
function NLPModels.hess_coord!(m::ExaNLPModel,x, l, hess::AbstractVector; obj_weight=1.0)
    m.timers.hessian_time += @elapsed begin
        hessian_lagrangian_coo!(m.nlp, hess, x, l, obj_weight)
    end
end

# Hessian: dense callback
function MadNLP.hess_dense!(m::ExaNLPModel, x, l, hess::AbstractMatrix; obj_weight=1.0)
    _update!(m, x)
    m.timers.hessian_time += @elapsed begin
        hessian_lagrangian!(m.nlp, hess, x, l, obj_weight)
    end
end

# Scaling
function MadNLP.scale_objective(m::ExaNLPModel{Ev}, g::AbstractVector, x::AbstractVector; max_scaling=1e-8) where {Ev<:ReducedSpaceEvaluator}
    return min(1.0, max_scaling / norm(m.nlp.grad, Inf))
end

function MadNLP.scale_constraints!(
    m::ExaNLPModel{Ev},
    con_scale::AbstractVector,
    jac::AbstractMatrix,
    x::AbstractVector;
    max_scaling=1e-8,
) where {Ev<:ReducedSpaceEvaluator}
    J = convert(SparseMatrixCSC, m.nlp.jac.J)
    m, n = size(J)
    for j in 1:n
        for c in J.colptr[j]:J.colptr[j+1]-1
            i = J.rowval[c]
            con_scale[i] = max(con_scale[i], abs(J.nzval[c]))
        end
    end
    con_scale .= min.(1.0, max_scaling ./ con_scale)
end

