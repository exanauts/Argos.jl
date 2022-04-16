mutable struct StochEvaluator{T, VI, VT, MT, JacCons, HessLag} <: AbstractNLPEvaluator
    model::ExaPF.PolarForm{T, VI, VT, MT}
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
    stack::ExaPF.BlockNetworkStack
    ∂stack::ExaPF.BlockNetworkStack

    jac::JacCons
    hess::HessLag

    map2tril::VI
end

function block_sparsity_hessian(H, nx, nu, nscen)
    i_hess, j_hess, _ = findnz(H)
    nnzh = length(i_hess)
    i_coo, j_coo = Int[], Int[]
    ind_uu = Tuple{Int, Int}[]
    idk = 1
    for (i, j) in zip(i_hess, j_hess)
        # Get current scenario
        k = div(i - 1, nx+nu) + 1
        @assert 1 <= k <= nscen
        ik = (i-1) % (nx + nu) + 1
        jk = j
        ## xx
        if (ik <= nx) && (jk <= nx)
            push!(i_coo, ik + (k-1) * nx)
            push!(j_coo, jk + (k-1) * nx)
        ## xu
        elseif (nx < ik <= nx+nu) && (jk <= nx)
            push!(i_coo, ik + (nscen-1) * nx)
            push!(j_coo, jk + (k-1) * nx)
        ## ux
        elseif (ik <= nx) && (nx < jk <= nx+nu)
            push!(i_coo, ik + (k-1) * nx)
            push!(j_coo, jk + (nscen-1) * nx)
        ## uu
        elseif (nx < ik <= nx+nu) && (nx < jk <= nx+nu)
            push!(i_coo, ik + (nscen-1) * nx)
            push!(j_coo, jk + (nscen-1) * nx)
            push!(ind_uu, (idk, k))
        end
        idk += 1
    end
    return i_coo, j_coo, ind_uu
end

function block_sparsity_jacobian(J, nx, nu, nscen, slices)
    i_jac, j_jac, _ = findnz(J)
    i_coo, j_coo = Int[], Int[]

    cumslices = nscen .* cumsum(slices)

    shift = 0
    idk = 1 # start with first constraints
    for (i, j) in zip(i_jac, j_jac)
        # Which constraint?
        idk = 1
        while i > cumslices[idk]
            idk += 1
        end
        shift = (idk > 1) ? cumslices[idk-1] : 0
        # Which scenario?
        k = div(i - shift - 1, slices[idk]) + 1
        @assert 0 < k <= nscen

        ## / x
        if j <= nx
            push!(i_coo, i)
            push!(j_coo, j + (k-1) * nx)
        ## / u
        elseif j <= nx + nu
            push!(i_coo, i)
            push!(j_coo, j + (nscen-1) * nx)
        end
    end
    return i_coo, j_coo
end

#=
    Ordering: [x1, x2, ..., xN, u]
=#
function StochEvaluator(
    model::PolarForm{T, VI, VT, MT}, ploads::Array{T, 2}, qloads::Array{T, 2};
    relax_eps=1e-6,
) where {T, VI, VT, MT}
    @assert size(ploads, 2) == size(qloads, 2)
    nscen = size(ploads, 2)

    nx = ExaPF.number(model, State())
    nu = ExaPF.number(model, Control())

    # Block mappings
    blk_mapxu = ExaPF.mapping(model, ExaPF.AllVariables(), nscen)
    blk_mapx = ExaPF.mapping(model, State(), nscen)
    blk_mapu = ExaPF.mapping(model, Control(), nscen)

    mapz = [blk_mapx; blk_mapu[1:nu]]

    # Expressions
    basis = ExaPF.PolarBasis(model)
    costs = ExaPF.CostFunction(model)
    constraints_expr = [
        ExaPF.PowerFlowBalance(model),
        ExaPF.PowerGenerationBounds(model),
        ExaPF.LineFlows(model),
    ]
    constraints = ExaPF.MultiExpressions(constraints_expr)
    m = length(constraints) * nscen

    stack = ExaPF.BlockNetworkStack(model, ploads, qloads)
    ∂stack = ExaPF.BlockNetworkStack(model, ploads, qloads)

    # Buffers
    obj = VT(undef, nscen)
    cons = VT(undef, m)
    grad_control = VT(undef, nu * nscen)
    y = VT(undef, m + nscen)

    s_min, s_max = ExaPF.bounds(model, stack)
    x_min, x_max = s_min[mapz], s_max[mapz]

    g_min = vcat([repeat(ExaPF.bounds(model, f)[1], nscen) for f in constraints_expr]...)
    g_max = vcat([repeat(ExaPF.bounds(model, f)[2], nscen) for f in constraints_expr]...)

    # Remove bounds below a given threshold
    g_max = min.(g_max, 1e5)
    # Remove equalities
    # TODO

    jac = ExaPF.ArrowheadJacobian(model, constraints ∘ basis, nscen)
    ExaPF.set_params!(jac, stack)
    lagrangian_expr = [costs; constraints_expr]
    lagrangian = ExaPF.MultiExpressions(lagrangian_expr)
    hess = ExaPF.ArrowheadHessian(model, lagrangian ∘ basis, nscen)
    println(typeof(hess))
    ExaPF.set_params!(hess, stack)

    map2tril = tril_mapping(hess.H)

    return StochEvaluator(
        model, nscen, nx, nu, blk_mapu, blk_mapx, mapz,
        basis, costs, constraints,
        obj, cons, grad_control, y, x_min, x_max, g_min, g_max,
        stack, ∂stack, jac, hess, map2tril,
    )
end
function StochEvaluator(datafile::String, pload, qload; device=ExaPF.CPU(), options...)
    return StochEvaluator(ExaPF.PolarForm(datafile, device), pload, qload; options...)
end

model(nlp::StochEvaluator) = nlp.model
backend(nlp::StochEvaluator) = nlp

n_variables(nlp::StochEvaluator) = nlp.nx * nlp.nscen + nlp.nu
n_constraints(nlp::StochEvaluator) = length(nlp.g_min)

constraints_type(::StochEvaluator) = :inequality
has_hessian(nlp::StochEvaluator) = true
has_hessian_lagrangian(nlp::StochEvaluator) = true

# Getters
Base.get(nlp::StochEvaluator, ::Constraints) = nlp.constraints
function Base.get(nlp::StochEvaluator, ::State)
    x = similar(nlp.x_min) ; fill!(x, 0)
    ExaPF.get!(nlp.model, State(), x, nlp.buffer)
    return x
end

# Physics
Base.get(nlp::StochEvaluator, ::PS.VoltageMagnitude) = nlp.stack.vmag
Base.get(nlp::StochEvaluator, ::PS.VoltageAngle) = nlp.stack.vang
Base.get(nlp::StochEvaluator, ::PS.ActivePower) = nlp.stack.pgen
function Base.get(nlp::StochEvaluator, attr::PS.AbstractNetworkAttribute)
    return ExaPF.get(nlp.model, attr)
end

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
    u = view(x, nlp.nscen*nlp.nx+1:nlp.nscen*nlp.nx+nlp.nu)
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

    gu = view(g, nlp.nscen*nlp.nx+1:nlp.nscen*nlp.nx+nlp.nu)
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
    copyto!(nlp._multipliers, nlp.nscen, y, 1, m)
    ExaPF.hessian!(nlp.hess, nlp.stack, nlp._multipliers)
    # TODO
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
    println(io, "    * constraints:")
end

function reset!(nlp::StochEvaluator)
    # Reset buffer
    fill!(nlp._obj, 0)
    fill!(nlp._multipliers, 0)
    fill!(nlp._cons, 0)
    empty!(nlp.stack)
    empty!(nlp.∂stack)
    ExaPF.init!(nlp.model, nlp.stack; loads=false)
    fill!(nonzeros(nlp.hess.H), 0)
    return
end

