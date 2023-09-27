mutable struct CorrectiveEvaluator{T, VI, VT, MT, JacCons, HessLag} <: AbstractNLPEvaluator
    model::ExaPF.PolarFormRecourse{T, VI, VT, MT}
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
function CorrectiveEvaluator(
    model::PolarForm{T, VI, VT, MT}, nscen::Int;
    contingencies=ExaPF.AbstractContingency[],
    relax_eps=1e-6, line_constraints=true, epsilon=1e-4,
    tracking=false, stack_ref=nothing,
) where {T, VI, VT, MT}
    blk_model = ExaPF.PolarFormRecourse(model, nscen)

    nx = ExaPF.number(blk_model, State())
    nu = ExaPF.number(blk_model, Control())

    # Block mappings
    blk_mapxu = ExaPF.mapping(blk_model, ExaPF.AllVariables(), nscen)
    blk_mapx = ExaPF.mapping(blk_model, State(), nscen)
    blk_mapu = ExaPF.mapping(blk_model, Control(), nscen)

    mapz = [blk_mapx; blk_mapu[1:nu]]

    # Expressions
    basis = ExaPF.PolarBasis(blk_model)
    costs = if tracking
        @assert isa(stack_ref, ExaPF.NetworkStack)
        ExaPF.TrackingCost(blk_model, stack_ref)
    else
        ExaPF.QuadraticCost(blk_model)
    end

    if length(contingencies) >= 1
        constraints_expr = Any[
            ExaPF.PowerFlowRecourse(blk_model, contingencies; epsilon=epsilon),
            ExaPF.ReactivePowerBounds(blk_model, contingencies),
            ExaPF.LineFlows(blk_model, contingencies),
        ]
    else
        constraints_expr = Any[
            ExaPF.PowerFlowRecourse(blk_model; epsilon=epsilon),
            ExaPF.ReactivePowerBounds(blk_model),
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

    # Remove bounds below a given threshold
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

    map2tril = tril_mapping(hess.H)

    return CorrectiveEvaluator(
        blk_model, nscen, nx*nscen, nu, VI(blk_mapu), VI(blk_mapx), VI(mapz),
        basis, costs, constraints,
        obj, cons, grad_control, y, x_min, x_max, g_min, g_max,
        stack, ∂stack, jac, hess, map2tril,
    )
end

function CorrectiveEvaluator(
    model::PolarForm,
    ploads::Array{Float64, 2},
    qloads::Array{Float64, 2};
    options...
)
    @assert size(ploads, 2) == size(qloads, 2)
    nscen = size(ploads, 2)
    evaluator = CorrectiveEvaluator(model, nscen; options...)
    # Set loads as parameters in the model
    ExaPF.set_params!(evaluator.stack, ploads, qloads)
    ExaPF.set_params!(evaluator.∂stack, ploads, qloads)
    ExaPF.set_params!(evaluator.jac, evaluator.stack)
    ExaPF.set_params!(evaluator.hess, evaluator.stack)
    return evaluator
end

function CorrectiveEvaluator(
    datafile::String,
    pload::Array{Float64, 2},
    qload::Array{Float64, 2};
    device=ExaPF.CPU(),
    options...
)
    return CorrectiveEvaluator(ExaPF.PolarForm(datafile, device), pload, qload; options...)
end

model(nlp::CorrectiveEvaluator) = nlp.model
backend(nlp::CorrectiveEvaluator) = nlp

n_variables(nlp::CorrectiveEvaluator) = nlp.nx + nlp.nu
n_constraints(nlp::CorrectiveEvaluator) = length(nlp.g_min)

constraints_type(::CorrectiveEvaluator) = :inequality
has_hessian(nlp::CorrectiveEvaluator) = true
has_hessian_lagrangian(nlp::CorrectiveEvaluator) = true

# Initial position
function initial(nlp::CorrectiveEvaluator{T,VI,VT,MT}) where {T,VI,VT,MT}
    x = VT(undef, n_variables(nlp))
    copyto!(x, nlp.stack, nlp.mapz)
    return x
end

# Bounds
bounds(nlp::CorrectiveEvaluator, ::Variables) = (nlp.x_min, nlp.x_max)
bounds(nlp::CorrectiveEvaluator, ::Constraints) = (nlp.g_min, nlp.g_max)

## Callbacks
# Update buffer with new state and control
function update!(nlp::CorrectiveEvaluator, x)
    copyto!(nlp.stack, nlp.mapz, x)
    u = view(x, nlp.nx+1:nlp.nx+nlp.nu)
    ExaPF.blockcopy!(nlp.stack, nlp.blk_mapu, u)
    # Full forward pass
    nlp.basis(nlp.stack.ψ, nlp.stack)
    nlp.costs(nlp._obj, nlp.stack)
    nlp.constraints(nlp._cons, nlp.stack)
    return true
end

objective(nlp::CorrectiveEvaluator, x) = 1.0 / nlp.nscen * sum(nlp._obj)
constraint!(nlp::CorrectiveEvaluator, c, x) = copyto!(c, nlp._cons)

###
# First-order code
####
function gradient!(nlp::CorrectiveEvaluator, g, x)
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

function jprod!(nlp::CorrectiveEvaluator, jv, x, v)
    ExaPF.jacobian!(nlp.jac, nlp.stack)
    mul!(jv, nlp.jac.J, v)
    return
end

function jtprod!(nlp::CorrectiveEvaluator, jv, x, v)
    ExaPF.jacobian!(nlp.jac, nlp.stack)
    mul!(jv, nlp.jac.J', v)
    return
end

function jacobian_structure(nlp::CorrectiveEvaluator)
    i, j, _ = findnz(nlp.jac.J)
    return i, j
end

function jacobian_coo!(nlp::CorrectiveEvaluator, jacval::AbstractVector, x)
    ExaPF.jacobian!(nlp.jac, nlp.stack)
    copyto!(jacval, nonzeros(nlp.jac.J))
    return
end

###
# Second-order code
####
function hessian_lagrangian_prod!(
    nlp::CorrectiveEvaluator, hessvec, x, y, σ, w,
)
    m = n_constraints(nlp)::Int
    nlp._multipliers[1:nlp.nscen] .= σ / nlp.nscen
    nlp._multipliers[nlp.nscen+1:m+nlp.nscen] .= y

    ExaPF.hessian!(nlp.hess, nlp.stack, nlp._multipliers)
    mul!(hessvec, nlp.hess.H, w)
    return
end

function hessian_lagrangian_coo!(nlp::CorrectiveEvaluator, hess, x, y, σ)
    n = n_variables(nlp)::Int
    m = n_constraints(nlp)
    nlp._multipliers[1:nlp.nscen] .= σ / nlp.nscen * 0.01
    copyto!(nlp._multipliers, nlp.nscen + 1, y, 1, m)
    ExaPF.hessian!(nlp.hess, nlp.stack, nlp._multipliers)
    # Keep only lower-triangular part
    transfer2tril!(hess, nlp.hess.H, nlp.map2tril)
    return
end

# Return lower-triangular matrix
function hessian_structure(nlp::CorrectiveEvaluator)
    i, j, _ = findnz(nlp.hess.H)
    rows = Int[ix for (ix, jx) in zip(i, j) if jx <= ix]
    cols = Int[jx for (ix, jx) in zip(i, j) if jx <= ix]
    return rows, cols
end

function Base.show(io::IO, nlp::CorrectiveEvaluator)
    n = n_variables(nlp)
    m = n_constraints(nlp)
    println(io, "A CorrectiveEvaluator object")
    println(io, "    * device: ", nlp.model.device)
    println(io, "    * #vars: ", n)
    println(io, "    * #cons: ", m)
end

function reset!(nlp::CorrectiveEvaluator)
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

