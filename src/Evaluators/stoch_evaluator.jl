mutable struct StochEvaluator{T, VI, VT, MT, JacCons, HessLag} <: AbstractNLPEvaluator
    model::ExaPF.BlockPolarForm{T, VI, VT, MT}
    nscen::Int
    nx::Int
    nu::Int
    blk_mapu::VI
    blk_mapx::VI
    mapz::VI

    # Expressions
    basis::AutoDiff.AbstractExpression
    costs::AutoDiff.AbstractExpression
    constraints::AutoDiff.AbstractExpression

    _obj::VT
    _cons::VT
    _grad_control::VT
    _multipliers::VT

    x_min::VT
    x_max::VT
    g_min::VT
    g_max::VT

    # Cache
    stack::ExaPF.NetworkStack
    ∂stack::ExaPF.NetworkStack

    jac::JacCons
    hess::HessLag

    map2tril::VI
end

#=
    Ordering: [x1, x2, ..., xN, u]
=#
function StochEvaluator(
    model::PolarForm{T, VI, VT, MT},
    nscen::Int;
    contingencies=ExaPF.AbstractContingency[],
    relax_eps=1e-6,
) where {T, VI, VT, MT}
    if length(contingencies) >= 1
        @assert nscen == length(contingencies) + 1
    end
    blk_model = ExaPF.BlockPolarForm(model, nscen)

    nx = ExaPF.number(model, State())
    nu = ExaPF.number(model, Control())

    # Block mappings
    blk_mapxu = ExaPF.mapping(model, ExaPF.AllVariables(), nscen)
    blk_mapx = ExaPF.mapping(model, State(), nscen)
    blk_mapu = ExaPF.mapping(model, Control(), nscen)

    mapz = [blk_mapx; blk_mapu[1:nu]]

    # Expressions
    basis = ExaPF.PolarBasis(blk_model)
    costs = ExaPF.CostFunction(blk_model)
    if length(contingencies) >= 1
        constraints_expr = [
            ExaPF.PowerFlowBalance(blk_model, contingencies),
            ExaPF.PowerGenerationBounds(blk_model, contingencies),
            ExaPF.LineFlows(blk_model, contingencies),
        ]
    else
        constraints_expr = [
            ExaPF.PowerFlowBalance(blk_model),
            ExaPF.PowerGenerationBounds(blk_model),
            ExaPF.LineFlows(blk_model),
        ]
    end
    constraints = ExaPF.MultiExpressions(constraints_expr)
    m = length(constraints)

    stack = ExaPF.NetworkStack(blk_model)
    ∂stack = ExaPF.NetworkStack(blk_model)

    # Buffers
    obj = VT(undef, nscen)
    cons = VT(undef, m)
    grad_control = VT(undef, nu * nscen)
    y = VT(undef, m + nscen)

    s_min, s_max = ExaPF.bounds(blk_model, stack)
    x_min, x_max = s_min[mapz], s_max[mapz]

    g_min, g_max = ExaPF.bounds(blk_model, constraints)
    # Remove bounds above a given threshold
    g_max = min.(g_max, 1e5)
    # Remove equalities
    ggl = @view g_min[nx*nscen+1:end]
    ggu = @view g_max[nx*nscen+1:end]
    idx_eq = findall(ggl .== ggu)
    if length(idx_eq) > 0
        println("[Argos] Elastic relaxation of $(length(idx_eq)) operational eq. constraints")
        ggu[idx_eq] .+= 1e-6
        ggl[idx_eq] .-= 1e-6
    end

    jac = ExaPF.ArrowheadJacobian(blk_model, constraints ∘ basis, ExaPF.AllVariables())
    lagrangian_expr = [costs; constraints_expr]
    lagrangian = ExaPF.MultiExpressions(lagrangian_expr)
    hess = ExaPF.ArrowheadHessian(blk_model, lagrangian ∘ basis, ExaPF.AllVariables())
    ExaPF.set_params!(hess, stack)

    map2tril = tril_mapping(hess.H)

    return StochEvaluator(
        blk_model, nscen, nx*nscen, nu, VI(blk_mapu), VI(blk_mapx), VI(mapz),
        basis, costs, constraints,
        obj, cons, grad_control, y, x_min, x_max, g_min, g_max,
        stack, ∂stack, jac, hess, map2tril,
    )
end
function StochEvaluator(
    datafile::String,
    pload::Array{Float64, 2},
    qload::Array{Float64, 2};
    device=ExaPF.CPU(),
    options...
)
    return StochEvaluator(ExaPF.PolarForm(datafile, device), pload, qload; options...)
end
function StochEvaluator(
    model::PolarForm{T, VI, VT, MT},
    ploads::Array{T, 2},
    qloads::Array{T, 2};
    options...
) where {T, VI, VT, MT}
    @assert size(ploads, 2) == size(qloads, 2)
    nscen = size(ploads, 2)
    evaluator = StochEvaluator(model, nscen; options...)
    # Set loads as parameters in the model
    ExaPF.set_params!(evaluator.stack, ploads, qloads)
    ExaPF.set_params!(evaluator.∂stack, ploads, qloads)
    ExaPF.set_params!(evaluator.jac, evaluator.stack)
    ExaPF.set_params!(evaluator.hess, evaluator.stack)
    return evaluator
end

model(nlp::StochEvaluator) = nlp.model
backend(nlp::StochEvaluator) = nlp

n_variables(nlp::StochEvaluator) = nlp.nx + nlp.nu
n_constraints(nlp::StochEvaluator) = length(nlp.g_min)

constraints_type(::StochEvaluator) = :inequality
has_hessian(nlp::StochEvaluator) = true
has_hessian_lagrangian(nlp::StochEvaluator) = true

# Initial position
function initial(nlp::StochEvaluator{T,VI,VT,MT}) where {T,VI,VT,MT}
    x = VT(undef, n_variables(nlp))
    copyto!(x, nlp.stack, nlp.mapz)
    return x
end

# Bounds
bounds(nlp::StochEvaluator, ::Variables) = (nlp.x_min, nlp.x_max)
bounds(nlp::StochEvaluator, ::Constraints) = (nlp.g_min, nlp.g_max)

## Callbacks
# Update buffer with new state and control
function update!(nlp::StochEvaluator, x)
    copyto!(nlp.stack, nlp.mapz, x)
    u = view(x, nlp.nx+1:nlp.nx+nlp.nu)
    ExaPF.blockcopy!(nlp.stack, nlp.blk_mapu, u)
    # Full forward pass
    nlp.basis(nlp.stack.ψ, nlp.stack)
    nlp.costs(nlp._obj, nlp.stack)
    nlp.constraints(nlp._cons, nlp.stack)
    return true
end

objective(nlp::StochEvaluator, x) = 1.0 / nlp.nscen * sum(nlp._obj)
constraint!(nlp::StochEvaluator, c, x) = copyto!(c, nlp._cons)

###
# First-order code
####
function gradient!(nlp::StochEvaluator, g, x)
    ExaPF.empty!(nlp.∂stack)
    ExaPF.adjoint!(nlp.costs, nlp.∂stack, nlp.stack, 1.0)
    ExaPF.adjoint!(nlp.basis, nlp.∂stack, nlp.stack, nlp.∂stack.ψ)
    copyto!(g, nlp.∂stack, nlp.mapz)
    copyto!(nlp._grad_control, nlp.∂stack, nlp.blk_mapu)

    gu = view(g, nlp.nx+1:nlp.nx+nlp.nu)
    sum!(gu, reshape(nlp._grad_control, nlp.nu, nlp.nscen))
    g .*= 1.0 / nlp.nscen
    return
end

function jprod!(nlp::StochEvaluator, jv, x, v)
    ExaPF.jacobian!(nlp.jac, nlp.stack)
    mul!(jv, nlp.jac.J, v)
    return
end

function jtprod!(nlp::StochEvaluator, jv, x, v)
    ExaPF.jacobian!(nlp.jac, nlp.stack)
    mul!(jv, nlp.jac.J', v)
    return
end

function jacobian_structure(nlp::StochEvaluator)
    i, j, _ = findnz(nlp.jac.J)
    return i, j
end

function jacobian_coo!(nlp::StochEvaluator, jacval::AbstractVector, x)
    ExaPF.jacobian!(nlp.jac, nlp.stack)
    copyto!(jacval, nonzeros(nlp.jac.J))
    return
end

###
# Second-order code
####
function hessian_lagrangian_prod!(
    nlp::StochEvaluator, hessvec, x, y, σ, w,
)
    m = n_constraints(nlp)::Int
    nlp._multipliers[1:nlp.nscen] .= σ / nlp.nscen
    nlp._multipliers[nlp.nscen+1:m+nlp.nscen] .= y

    ExaPF.hessian!(nlp.hess, nlp.stack, nlp._multipliers)
    mul!(hessvec, nlp.hess.H, w)
    return
end

function hessian_lagrangian_coo!(nlp::StochEvaluator, hess, x, y, σ)
    n = n_variables(nlp)::Int
    m = n_constraints(nlp)
    nlp._multipliers[1:nlp.nscen] .= σ / nlp.nscen
    copyto!(nlp._multipliers, nlp.nscen + 1, y, 1, m)
    ExaPF.hessian!(nlp.hess, nlp.stack, nlp._multipliers)
    # Keep only lower-triangular part
    transfer2tril!(hess, nlp.hess.H, nlp.map2tril)
    return
end

# Return lower-triangular matrix
function hessian_structure(nlp::StochEvaluator)
    i, j, _ = findnz(nlp.hess.H)
    rows = Int[ix for (ix, jx) in zip(i, j) if jx <= ix]
    cols = Int[jx for (ix, jx) in zip(i, j) if jx <= ix]
    return rows, cols
end

function Base.show(io::IO, nlp::StochEvaluator)
    n = n_variables(nlp)
    m = n_constraints(nlp)
    println(io, "A StochEvaluator object")
    println(io, "    * device: ", nlp.model.device)
    println(io, "    * #vars: ", n)
    println(io, "    * #cons: ", m)
end

function reset!(nlp::StochEvaluator)
    # Reset buffer
    fill!(nlp._obj, 0)
    fill!(nlp._multipliers, 0)
    fill!(nlp._cons, 0)
    empty!(nlp.stack)
    empty!(nlp.∂stack)
    ExaPF.init!(nlp.model, nlp.stack; update_loads=false)
    fill!(nonzeros(nlp.hess.H), 0)
    return
end

