struct OPFModel{Evaluator} <: NLPModels.AbstractNLPModel{Float64,Vector{Float64}}
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
    etc::Dict{Symbol, Any}
end

function OPFModel(nlp::AbstractNLPEvaluator)
    n = n_variables(nlp)
    m = n_constraints(nlp)

    # Initial variable
    y0 = zeros(m)
    # Bounds
    xl, xu = bounds(nlp, Variables())
    gl, gu = bounds(nlp, Constraints())
    x0  = initial(nlp)
    # Sparsity
    hrows, hcols = hessian_structure(nlp)
    jrows, jcols = jacobian_structure(nlp)
    nnzh = length(hrows)
    nnzj = length(jrows)

    etc = Dict{Symbol, Any}()

    return OPFModel{typeof(nlp)}(
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
        hrows, hcols, jrows, jcols, etc,
    )
end

backend(m::OPFModel) = backend(m.nlp)

function NLPModels.jac_structure!(m::OPFModel, rows::AbstractVector, cols::AbstractVector)
    copyto!(rows, m.jrows)
    copyto!(cols, m.jcols)
    return rows, cols
end

function NLPModels.hess_structure!(m::OPFModel, rows::AbstractVector, cols::AbstractVector)
    copyto!(rows, m.hrows)
    copyto!(cols, m.hcols)
    return rows, cols
end

function _update!(m::OPFModel, x::AbstractVector)
    hx = hash(x)
    if hx != m.hash_x[1]
        m.timers.update_time += @elapsed begin
            update!(m.nlp, x)
        end
        m.hash_x[1] = hx
    end
end

# Objective
function NLPModels.obj(m::OPFModel,x::AbstractVector)
    _update!(m, x)
    m.timers.obj_time += @elapsed begin
        obj = objective(m.nlp, x)
    end
    return obj
end

# Gradient
function NLPModels.grad!(m::OPFModel, x::AbstractVector, g::AbstractVector)
    _update!(m, x)
    m.timers.grad_time += @elapsed begin
        gradient!(m.nlp, g, x)
    end
    return g
end

# Constraints
function NLPModels.cons!(m::OPFModel,x::AbstractVector,c::AbstractVector)
    _update!(m, x)
    m.timers.cons_time += @elapsed begin
        constraint!(m.nlp, c, x)
    end
    return c
end

# Jacobian: sparse callback
function NLPModels.jac_coord!(m::OPFModel, x::AbstractVector, jac::AbstractVector)
    _update!(m, x)
    nnzj = NLPModels.get_nnzj(m)
    jv = view(jac, 1:nnzj) # NB: ensure compatibility with MadNLP
    m.timers.jacobian_time += @elapsed begin
        jacobian_coo!(m.nlp, jv, x)
    end
end

function NLPModels.jtprod!(m::OPFModel, x::AbstractVector, v::AbstractVector, jv::AbstractVector)
    _update!(m, x)
    jtprod!(m.nlp, jv, x, v)
end

# Jacobian: dense callback
function MadNLP.jac_dense!(m::OPFModel, x, J::AbstractMatrix)
    _update!(m, x)
    m.timers.jacobian_time += @elapsed begin
        jacobian!(m.nlp, J, x)
    end
end

# Hessian-vector products
function NLPModels.hprod!(m::OPFModel, x::AbstractVector, l::AbstractVector, v::AbstractVector, hv::AbstractVector; obj_weight=1.0)
    m.timers.hessprod_time += @elapsed begin
        hessian_lagrangian_prod!(m.nlp, hv, x, l, obj_weight, v)
    end
end

# Hessian: sparse callback
function NLPModels.hess_coord!(m::OPFModel,x::AbstractVector, l::AbstractVector, hess::AbstractVector; obj_weight=1.0)
    m.timers.hessian_time += @elapsed begin
        hessian_lagrangian_coo!(m.nlp, hess, x, l, obj_weight)
    end
end

# Hessian: dense callback
function MadNLP.hess_dense!(m::OPFModel, x, l, hess::AbstractMatrix; obj_weight=1.0)
    _update!(m, x)
    m.timers.hessian_time += @elapsed begin
        hessian_lagrangian!(m.nlp, hess, x, l, obj_weight)
    end
end

# ## Note
# When we are working in the reduced space, we modify the scaling
# of the problem to avoid depending on the inverse of the Jacobian Gx.

function MadNLP.scale_objective(m::OPFModel, grad::AbstractVector; max_gradient=1e-8)
    # nlp = backend(m)
    # g_ = isa(nlp, ReducedSpaceEvaluator) ? nlp.grad : grad
    return 1e-4 #min(1, max_gradient / norm(g_, Inf))
end

function MadNLP.scale_constraints!(
    m::OPFModel,
    con_scale::AbstractVector,
    jac::AbstractMatrix;
    max_gradient=1e-8,
)
    # nlp = backend(m)
    # J = nlp.jac.J
    # fill!(con_scale, 0.0)
    # Ji, Jj, Jv = SparseArrays.findnz(J)
    # for i in 1:SparseArrays.nnz(J)
    #     row = @inbounds Ji[i]
    #     @inbounds con_scale[row] = max(con_scale[row], abs(Jv[i]))
    # end
    con_scale .= 1e-2 #min.(1.0, max_gradient ./ con_scale)
end

