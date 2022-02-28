
mutable struct FullSpaceEvaluator{T, VI, VT, MT, JacCons, HessLag} <: AbstractNLPEvaluator
    model::ExaPF.PolarForm{T, VI, VT, MT}
    nx::Int
    nu::Int
    mapxu::VI

    # Expressions
    basis::AutoDiff.AbstractExpression
    costs::AutoDiff.AbstractExpression
    constraints::AutoDiff.AbstractExpression

    _obj::T
    _cons::VT
    _multipliers::VT

    x_min::VT
    x_max::VT
    g_min::VT
    g_max::VT

    # Cache
    stack::ExaPF.ExaPF.NetworkStack
    ∂stack::ExaPF.ExaPF.NetworkStack

    jac::JacCons
    hess::HessLag
    map2tril::VI
end

function FullSpaceEvaluator(
    model::PolarForm{T, VI, VT, MT};
    line_constraints=true,
) where {T, VI, VT, MT}
    # Load mapping
    mapx = ExaPF.my_map(model, State())
    mapu = ExaPF.my_map(model, Control())
    mapxu = [mapx; mapu]
    nx = length(mapx)
    nu = length(mapu)

    # Expressions
    basis = ExaPF.PolarBasis(model)
    costs = ExaPF.CostFunction(model)
    constraints_expr = [
        ExaPF.PowerFlowBalance(model),
        ExaPF.PowerGenerationBounds(model),
    ]
    if line_constraints
        push!(constraints_expr, ExaPF.LineFlows(model))
    end
    constraints = ExaPF.MultiExpressions(constraints_expr)
    m = length(constraints)

    stack = ExaPF.NetworkStack(model)
    ∂stack = ExaPF.NetworkStack(model)
    # Buffers
    obj = Inf
    cons = VT(undef, m)
    y = VT(undef, m + 1)

    s_min, s_max = ExaPF.bounds(model, stack)
    x_min, x_max = s_min[mapxu], s_max[mapxu]

    g_min, g_max = ExaPF.bounds(model, constraints)
    # Remove bounds below a given threshold
    g_max = min.(g_max, 1e5)
    # Remove equalities
    ggl = @view g_min[nx+1:end]
    ggu = @view g_max[nx+1:end]
    idx_eq = findall(ggl .== ggu)
    if length(idx_eq) > 0
        println("eq found")
        ggu[idx_eq] .+= 1e-6
        ggl[idx_eq] .-= 1e-6
    end

    jac = ExaPF.Jacobian(model, constraints ∘ basis, mapxu)
    lagrangian_expr = [costs; constraints_expr]
    lagrangian = ExaPF.MultiExpressions(lagrangian_expr)
    hess = ExaPF.FullHessian(model, lagrangian ∘ basis, mapxu)
    nonzeros(hess.H) .= 0.0 # TODO: do we need it ?

    mapxu = mapxu |> VI
    map2tril = tril_mapping(hess.H)

    return FullSpaceEvaluator(
        model, nx, nu, mapxu,
        basis, costs, constraints,
        obj, cons, y, x_min, x_max, g_min, g_max,
        stack, ∂stack, jac, hess, map2tril,
    )
end
function FullSpaceEvaluator(datafile::String; device=ExaPF.CPU(), options...)
    return FullSpaceEvaluator(ExaPF.PolarForm(datafile, device); options...)
end

model(nlp::FullSpaceEvaluator) = nlp.model
backend(nlp::FullSpaceEvaluator) = nlp

n_variables(nlp::FullSpaceEvaluator) = nlp.nx + nlp.nu
n_constraints(nlp::FullSpaceEvaluator) = length(nlp.g_min)

constraints_type(::FullSpaceEvaluator) = :inequality
has_hessian(nlp::FullSpaceEvaluator) = true
has_hessian_lagrangian(nlp::FullSpaceEvaluator) = true

# Getters
Base.get(nlp::FullSpaceEvaluator, ::Constraints) = nlp.constraints
function Base.get(nlp::FullSpaceEvaluator, ::State)
    x = similar(nlp.x_min) ; fill!(x, 0)
    ExaPF.get!(nlp.model, State(), x, nlp.buffer)
    return x
end

# Physics
Base.get(nlp::FullSpaceEvaluator, ::PS.VoltageMagnitude) = nlp.stack.vmag
Base.get(nlp::FullSpaceEvaluator, ::PS.VoltageAngle) = nlp.stack.vang
Base.get(nlp::FullSpaceEvaluator, ::PS.ActivePower) = nlp.stack.pgen
function Base.get(nlp::FullSpaceEvaluator, attr::PS.AbstractNetworkAttribute)
    return ExaPF.get(nlp.model, attr)
end

# Initial position
function initial(nlp::FullSpaceEvaluator{T,VI,VT,MT}) where {T,VI,VT,MT}
    x = VT(undef, n_variables(nlp))
    copyto!(x, nlp.stack, nlp.mapxu)
    return x
end

# Bounds
bounds(nlp::FullSpaceEvaluator, ::Variables) = nlp.x_min, nlp.x_max
bounds(nlp::FullSpaceEvaluator, ::Constraints) = (nlp.g_min, nlp.g_max)

## Callbacks
# Update buffer with new state and control
function update!(nlp::FullSpaceEvaluator, x)
    copyto!(nlp.stack, nlp.mapxu, x)
    # Full forward pass
    nlp.basis(nlp.stack.ψ, nlp.stack)
    nlp._obj = sum(nlp.costs(nlp.stack))
    nlp.constraints(nlp._cons, nlp.stack)
    return true
end

objective(nlp::FullSpaceEvaluator, x) = nlp._obj
constraint!(nlp::FullSpaceEvaluator, c, x) = copyto!(c, nlp._cons)

###
# First-order code
####
function gradient!(nlp::FullSpaceEvaluator, g, x)
    ExaPF.empty!(nlp.∂stack)
    ExaPF.adjoint!(nlp.costs, nlp.∂stack, nlp.stack, 1.0)
    ExaPF.adjoint!(nlp.basis, nlp.∂stack, nlp.stack, nlp.∂stack.ψ)
    copyto!(g, nlp.∂stack, nlp.mapxu)
    return
end

function jprod!(nlp::FullSpaceEvaluator, jv, x, v)
    ExaPF.jacobian!(nlp.jac, nlp.stack)
    vx = view(v, 1:nlp.nx)
    vu = view(v, nlp.nx+1:nlp.nx+nlp.nu)
    mul!(jv, nlp.jac.J, v)
    return
end

function jtprod!(nlp::FullSpaceEvaluator, jv, x, v)
    ExaPF.jacobian!(nlp.jac, nlp.stack)
    J = nlp.jac.J
    mul!(jv, J', v)
    return
end

function jacobian_structure(nlp::FullSpaceEvaluator)
    J = nlp.jac.J
    i, j, _ = findnz(J)
    return i, j
end

function jacobian_coo!(nlp::FullSpaceEvaluator, jacval::AbstractVector, x)
    ExaPF.jacobian!(nlp.jac, nlp.stack)
    J = nlp.jac.J
    copyto!(jacval, nonzeros(J))
    return
end

###
# Second-order code
####
function hessian_lagrangian_prod!(
    nlp::FullSpaceEvaluator, hessvec, x, y, σ, w,
)
    m = n_constraints(nlp)::Int
    nlp._multipliers[1] = σ
    nlp._multipliers[2:m+1] .= y

    ExaPF.hessian!(nlp.hess, nlp.stack, nlp._multipliers)
    H = nlp.hess.H
    mul!(hessvec, H, w)
    return
end

function hessian_lagrangian_coo!(nlp::FullSpaceEvaluator, hess, x, y, σ)
    n = n_variables(nlp)::Int
    m = n_constraints(nlp)
    nlp._multipliers[1:1] .= σ
    copyto!(nlp._multipliers, 2, y, 1, m)
    ExaPF.hessian!(nlp.hess, nlp.stack, nlp._multipliers)
    # Keep only lower-triangular part
    transfer2tril!(hess, nlp.hess.H, nlp.map2tril)
    return
end

# Return lower-triangular matrix
function hessian_structure(nlp::FullSpaceEvaluator)
    i, j, _ = findnz(nlp.hess.H)
    rows = Int[ix for (ix, jx) in zip(i, j) if jx <= ix]
    cols = Int[jx for (ix, jx) in zip(i, j) if jx <= ix]
    return rows, cols
end

function Base.show(io::IO, nlp::FullSpaceEvaluator)
    n = n_variables(nlp)
    m = n_constraints(nlp)
    println(io, "A FullSpaceEvaluator object")
    println(io, "    * device: ", nlp.model.device)
    println(io, "    * #vars: ", n)
    println(io, "    * #cons: ", m)
    println(io, "    * constraints:")
end

function reset!(nlp::FullSpaceEvaluator)
    # Reset buffer
    nlp._obj = Inf
    fill!(nlp._multipliers, 0)
    fill!(nlp._cons, 0)
    empty!(nlp.stack)
    empty!(nlp.∂stack)
    ExaPF.init!(nlp.model, nlp.stack)

    fill!(nonzeros(nlp.hess.H), 0)
    return
end

