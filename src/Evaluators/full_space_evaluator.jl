
mutable struct FullSpaceEvaluator{T, VI, VT, MT, JacCons, HessLag} <: AbstractNLPEvaluator
    model::ExaPF.PolarForm{T, VI, VT, MT}
    nx::Int
    nu::Int
    x_min::VT
    x_max::VT
    u_min::VT
    u_max::VT

    constraints::Vector{Function}
    g_min::VT
    g_max::VT

    # Cache
    buffer::ExaPF.PolarNetworkState{VI, VT}
    # AutoDiff
    obj_stack::AutoDiff.TapeMemory{typeof(ExaPF.cost_production), ExaPF.AdjointStackObjective{VT}, Nothing}
    cons_stacks::Vector{AutoDiff.TapeMemory} # / constraints
    constraint_jacobians::JacCons
    hesslag::HessLag
end

function FullSpaceEvaluator(
    model::PolarForm{T, VI, VT, MT};
    line_constraints=false,
) where {T, VI, VT, MT}
    # First, build up a network buffer
    buffer = ExaPF.get(model, ExaPF.PhysicalState())
    # Populate buffer with default values of the network, as stored inside model
    ExaPF.init_buffer!(model, buffer)

    u_min, u_max = ExaPF.bounds(model, Control())
    x_min, x_max = ExaPF.bounds(model, State())

    constraints = Function[
        ExaPF.power_balance,
        ExaPF.active_power_constraints,
        ExaPF.reactive_power_constraints
    ]
    if line_constraints
        push!(constraints, ExaPF.flow_constraints)
    end

    m = sum(Int[ExaPF.size_constraint(model, cons)::Int for cons in constraints])
    g_min = VT(undef, m)
    g_max = VT(undef, m)

    shift = 1
    for cons in constraints
        cb, cu = ExaPF.bounds(model, cons)
        m = ExaPF.size_constraint(model, cons)
        copyto!(g_min, shift, cb, 1, m)
        copyto!(g_max, shift, cu, 1, m)
        shift += m
    end

    obj_ad = ExaPF.pullback_objective(model)
    cons_ad = AutoDiff.TapeMemory[]
    for cons in constraints
        push!(cons_ad, AutoDiff.TapeMemory(model, cons, VT))
    end

    # Init Jacobian + coloring.
    # All results in CSC format.
    cons_jac = ExaPF.ConstraintsJacobianStorage(model, constraints)

    # Hessians
    m = length(g_min)
    func = line_constraints ? ExaPF.network_line_operations : ExaPF.network_operations
    hess_ad = FullHessianLagrangian(model, func, buffer)

    nx = length(x_min)
    nu = length(u_min)

    return FullSpaceEvaluator(
        model, nx, nu, x_min, x_max, u_min, u_max,
        constraints, g_min, g_max,
        buffer,
        obj_ad, cons_ad, cons_jac, hess_ad,
    )
end
function FullSpaceEvaluator(datafile::String; device=ExaPF.CPU(), options...)
    return FullSpaceEvaluator(ExaPF.PolarForm(datafile, device); options...)
end

backend(nlp::FullSpaceEvaluator) = nlp.model
inner_evaluator(nlp::FullSpaceEvaluator) = nlp

n_variables(nlp::FullSpaceEvaluator) = length(nlp.u_min) + length(nlp.x_min)
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
Base.get(nlp::FullSpaceEvaluator, ::ExaPF.PhysicalState) = nlp.buffer

# Physics
Base.get(nlp::FullSpaceEvaluator, ::PS.VoltageMagnitude) = nlp.buffer.vmag
Base.get(nlp::FullSpaceEvaluator, ::PS.VoltageAngle) = nlp.buffer.vang
Base.get(nlp::FullSpaceEvaluator, ::PS.ActivePower) = nlp.buffer.pgen
Base.get(nlp::FullSpaceEvaluator, ::PS.ReactivePower) = nlp.buffer.qgen
function Base.get(nlp::FullSpaceEvaluator, attr::PS.AbstractNetworkAttribute)
    return ExaPF.get(nlp.model, attr)
end
get_nnzh(nlp::FullSpaceEvaluator) = length(nlp.hesslag.h_V)

# Setters
function setvalues!(nlp::FullSpaceEvaluator, attr::PS.AbstractNetworkValues, values)
    ExaPF.setvalues!(nlp.model, attr, values)
end
function setvalues!(nlp::FullSpaceEvaluator, attr::PS.ActiveLoad, values)
    ExaPF.setvalues!(nlp.buffer, attr, values)
end
function setvalues!(nlp::FullSpaceEvaluator, attr::PS.ReactiveLoad, values)
    ExaPF.setvalues!(nlp.buffer, attr, values)
end

# Transfer network values inside buffer
function transfer!(
    nlp::FullSpaceEvaluator, vm, va, pg, qg,
)
    setvalues!(nlp.buffer, PS.VoltageMagnitude(), vm)
    setvalues!(nlp.buffer, PS.VoltageAngle(), va)
    setvalues!(nlp.buffer, PS.ActivePower(), pg)
    setvalues!(nlp.buffer, PS.ReactivePower(), qg)
end

# Initial position
function initial(nlp::FullSpaceEvaluator)
    x = similar(nlp.x_min) ; fill!(x, 0.0)
    ExaPF.get!(nlp.model, State(), x, nlp.buffer)
    u = similar(nlp.u_min) ; fill!(u, 0.0)
    ExaPF.get!(nlp.model, Control(), u, nlp.buffer)
    return [x; u]
end

# Bounds
bounds(nlp::FullSpaceEvaluator, ::Variables) = ([nlp.x_min; nlp.u_min], [nlp.x_max; nlp.u_max])
bounds(nlp::FullSpaceEvaluator, ::Constraints) = (nlp.g_min, nlp.g_max)

## Callbacks
# Update buffer with new state and control
function update!(nlp::FullSpaceEvaluator, x)
    x_ = view(x, 1:nlp.nx)
    u_ = view(x, 1+nlp.nx:nlp.nx+nlp.nu)
    ExaPF.transfer_state!(nlp.model, nlp.buffer, x_)
    ExaPF.transfer!(nlp.model, nlp.buffer, u_)
    return true
end

function objective(nlp::FullSpaceEvaluator, x)
    return ExaPF.cost_production(nlp.model, nlp.buffer)
end

function constraint!(nlp::FullSpaceEvaluator, c, x)
    ϕ = nlp.buffer
    mf = 1::Int
    mt = 0::Int
    for cons in nlp.constraints
        m_ = ExaPF.size_constraint(nlp.model, cons)::Int
        mt += m_
        cons_ = @view(c[mf:mt])
        cons(nlp.model, cons_, ϕ)
        mf += m_
    end
end

###
# First-order code
####
function gradient!(nlp::FullSpaceEvaluator, g, x)
    buffer = nlp.buffer
    # Evaluate adjoint of cost function and update inplace AdjointStackObjective
    ExaPF.gradient_objective!(nlp.model, nlp.obj_stack, buffer)
    ∇fₓ, ∇fᵤ = nlp.obj_stack.stack.∇fₓ, nlp.obj_stack.stack.∇fᵤ
    copyto!(g, 1, ∇fₓ, 1, nlp.nx)
    copyto!(g, nlp.nx + 1, ∇fᵤ, 1, nlp.nu)
    return
end

function jprod!(nlp::FullSpaceEvaluator, jv, x, v)
    ExaPF.update_full_jacobian!(nlp.model, nlp.constraint_jacobians, nlp.buffer)
    vx = view(v, 1:nlp.nx)
    vu = view(v, nlp.nx+1:nlp.nx+nlp.nu)

    Jx = nlp.constraint_jacobians.Jx
    Ju = nlp.constraint_jacobians.Ju
    mul!(jv, Jx, vx)
    mul!(jv, Ju, vu, 1.0, 1.0)
    return
end

function jtprod!(nlp::FullSpaceEvaluator, jv, x, v)
    ExaPF.update_full_jacobian!(nlp.model, nlp.constraint_jacobians, nlp.buffer)
    jx = view(jv, 1:nlp.nx)
    ju = view(jv, nlp.nx+1:nlp.nx+nlp.nu)

    Jx = nlp.constraint_jacobians.Jx
    Ju = nlp.constraint_jacobians.Ju
    mul!(jx, Jx', v)
    mul!(ju, Ju', v)
    return
end

function jacobian_structure(nlp::FullSpaceEvaluator)
    Jx = nlp.constraint_jacobians.Jx
    Ju = nlp.constraint_jacobians.Ju
    J = [Jx Ju]
    i, j, _ = findnz(J)
    return i, j
end

function jacobian_coo!(nlp::FullSpaceEvaluator, jac::AbstractVector, x)
    ExaPF.update_full_jacobian!(nlp.model, nlp.constraint_jacobians, nlp.buffer)
    Jxv = nlp.constraint_jacobians.Jx.nzval
    Juv = nlp.constraint_jacobians.Ju.nzval
    nnjx = length(Jxv)
    nnju = length(Juv)
    copyto!(jac, 1, Jxv, 1, nnjx)
    copyto!(jac, nnjx+1, Juv, 1, nnju)
    return
end

###
# Second-order code
####
function set_multipliers!(nlp::FullSpaceEvaluator, y, σ)
    nx, nu = nlp.nx, nlp.nu
    nbus = ExaPF.get(nlp.model, ExaPF.PowerSystem.NumberOfBuses())

    μ = nlp.hesslag.y
    fill!(μ, 0.0)
    # power flow
    copyto!(μ, 1, y, 1, 2*nbus)
    # objective
    μ[2*nbus+1] = σ
    # line flow
    if nlp.constraints[end] == ExaPF.flow_constraints
        m = ExaPF.size_constraint(nlp.model, ExaPF.flow_constraints)::Int
        copyto!(μ, 2*nbus+2, y, 2*nbus+1, m)
    end
end
#

function hessian_lagrangian_prod!(
    nlp::FullSpaceEvaluator, hessvec, x, y, σ, w,
)
    H = nlp.hesslag
    fill!(hessvec, 0.0)
    set_multipliers!(nlp, y, σ)

    AutoDiff.adj_hessian_prod!(nlp.model, H.hess, hessvec, nlp.buffer, H.y, w)
    return
end

function hessian_lagrangian_coo!(nlp::FullSpaceEvaluator, hess, x, y, σ)
    hl = nlp.hesslag
    fill!(hess, 0.0)
    set_multipliers!(nlp, y, σ)

    μ, w, hv = hl.y, hl.tgt, hl.hv

    # Evaluate Hessian matrix with coloring vectors
    for i in 1:hl.ncolors
        copy!(w, hl.seeds[:, i])
        fill!(hv, 0.0)
        AutoDiff.adj_hessian_prod!(nlp.model, hl.hess, hv, nlp.buffer, μ, w)
        hl.compressedH[:, i] .= hv
    end

    # Uncompress Hessian
    k = 1
    for (i, j) in zip(hl.h_I, hl.h_J)
        hess[k] = 0.5 * (hl.compressedH[j, hl.coloring[i]] + hl.compressedH[i, hl.coloring[j]])
        k += 1
    end
    return
end

# Return lower-triangular matrix
function hessian_structure(nlp::FullSpaceEvaluator)
    rows = nlp.hesslag.h_I
    cols = nlp.hesslag.h_J
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
    for cons in nlp.constraints
        println(io, "        - ", cons)
    end
end

function reset!(nlp::FullSpaceEvaluator)
    # Reset buffer
    ExaPF.init_buffer!(nlp.model, nlp.buffer)
    return
end

