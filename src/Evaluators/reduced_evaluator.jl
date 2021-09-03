
"""
    ReducedSpaceEvaluator{T, VI, VT, MT, Jacx, Jacu, JacCons, Hess} <: AbstractNLPEvaluator

Reduced-space evaluator projecting the optimization problem
into the powerflow manifold defined by the nonlinear equation ``g(x, u) = 0``.
The state ``x`` is defined implicitly, as a function of the control
``u``. Hence, the powerflow equation is implicitly satisfied
when we are using this evaluator.

Once a new point `u` is passed to the evaluator,
the user needs to call the method `update!` to find the corresponding
state ``x(u)`` satisfying the balance equation ``g(x(u), u) = 0``.

Taking as input an `ExaPF.PolarForm` structure, the reduced evaluator
builds the bounds corresponding to the control `u`,
The reduced evaluator could be instantiated on the host memory, or on a specific device
(currently, only CUDA is supported).

## Examples

```julia-repl
julia> datafile = "case9.m"  # specify a path to a MATPOWER instance
julia> nlp = ReducedSpaceEvaluator(datafile)
A ReducedSpaceEvaluator object
    * device: KernelAbstractions.CPU()
    * #vars: 5
    * #cons: 10
    * constraints:
        - voltage_magnitude_constraints
        - active_power_constraints
        - reactive_power_constraints
    * linear solver: ExaPF.LinearSolvers.DirectSolver()
```

If a GPU is available, we could instantiate `nlp` as

```julia-repl
julia> nlp_gpu = ReducedSpaceEvaluator(datafile; device=CUDADevice())
A ReducedSpaceEvaluator object
    * device: KernelAbstractions.CUDADevice()
    * #vars: 5
    * #cons: 10
    * constraints:
        - voltage_magnitude_constraints
        - active_power_constraints
        - reactive_power_constraints
    * linear solver: ExaPF.LinearSolvers.DirectSolver()

```

## Note
Mathematically, we set apart the state ``x`` from the control ``u``,
and use a third variable ``y`` --- the by-product --- to denote the remaining
values of the network.
In the implementation of `ReducedSpaceEvaluator`,
we only deal with a control `u` and an attribute `buffer`,
storing all the physical values needed to describe the network.
The attribute `buffer` stores the values of the control `u`, the state `x`
and the by-product `y`. Each time we are calling the method `update!`,
the values of the control are copied into the buffer.

"""
mutable struct ReducedSpaceEvaluator{T, VI, VT, MT, Jacx, Jacu, JacCons, HessLag} <: AbstractNLPEvaluator
    model::ExaPF.PolarForm{T, VI, VT, MT}
    λ::VT

    u_min::VT
    u_max::VT

    constraints::Vector{Function}
    g_min::VT
    g_max::VT

    # Cache
    buffer::ExaPF.PolarNetworkState{VI, VT}
    # AutoDiff
    state_jacobian::ExaPF.FullSpaceJacobian{Jacx, Jacu}
    obj_stack::AutoDiff.TapeMemory{typeof(ExaPF.cost_production), ExaPF.AdjointStackObjective{VT}, Nothing}
    cons_stacks::Vector{AutoDiff.TapeMemory} # / constraints
    constraint_jacobians::JacCons
    hesslag::HessLag

    # Options
    linear_solver::LS.AbstractLinearSolver
    backward_solver::LS.AbstractLinearSolver
    powerflow_solver::ExaPF.AbstractNonLinearSolver
    has_jacobian::Bool
    update_jacobian::Bool
    has_hessian::Bool
end

function ReducedSpaceEvaluator(
    model::PolarForm{T, VI, VT, MT};
    line_constraints=false,
    linear_solver=nothing,
    backward_solver=nothing,
    powerflow_solver=NewtonRaphson(tol=1e-12),
    want_jacobian=true,
    nbatch_hessian=1,
) where {T, VI, VT, MT}
    # First, build up a network buffer
    buffer = ExaPF.get(model, ExaPF.PhysicalState())
    # Populate buffer with default values of the network, as stored
    # inside model
    ExaPF.init_buffer!(model, buffer)

    u_min, u_max = ExaPF.bounds(model, Control())
    λ = similar(buffer.dx)

    constraints = Function[
        ExaPF.voltage_magnitude_constraints,
        ExaPF.active_power_constraints,
        ExaPF.reactive_power_constraints
    ]
    if line_constraints
        push!(constraints, ExaPF.flow_constraints)
    end

    m = sum([ExaPF.size_constraint(model, cons) for cons in constraints])
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

    # Build Linear Algebra
    J = ExaPF.powerflow_jacobian_device(model)
    _linear_solver = isnothing(linear_solver) ? LS.DirectSolver(J) : linear_solver
    # Work because J has a symmetric sparsity structure
    _backward_solver = isnothing(backward_solver) ? LS.DirectSolver(J) : backward_solver

    obj_ad = ExaPF.pullback_objective(model)
    state_ad = ExaPF.FullSpaceJacobian(model, ExaPF.power_balance)
    cons_ad = AutoDiff.TapeMemory[]
    for cons in constraints
        push!(cons_ad, AutoDiff.TapeMemory(model, cons, VT))
    end

    # Jacobians
    cons_jac = nothing
    if want_jacobian
        cons_jac = ExaPF.ConstraintsJacobianStorage(model, constraints)
    end

    # Hessians
    want_hessian = (nbatch_hessian > 0)
    hess_ad = nothing
    if want_hessian
        func = line_constraints ? ExaPF.network_line_operations : ExaPF.network_operations
        hess_ad = if nbatch_hessian > 1
            BatchHessianLagrangian(model, func, J, nbatch_hessian)
        else
            HessianLagrangian(model, func, J)
        end
    end

    return ReducedSpaceEvaluator(
        model, λ, u_min, u_max,
        constraints, g_min, g_max,
        buffer,
        state_ad, obj_ad, cons_ad, cons_jac, hess_ad,
        _linear_solver, _backward_solver,
        powerflow_solver, want_jacobian, false, want_hessian,
    )
end
function ReducedSpaceEvaluator(datafile::String; device=ExaPF.CPU(), options...)
    return ReducedSpaceEvaluator(ExaPF.PolarForm(datafile, device); options...)
end

backend(nlp::ReducedSpaceEvaluator) = nlp.model
inner_evaluator(nlp::ReducedSpaceEvaluator) = nlp

n_variables(nlp::ReducedSpaceEvaluator) = length(nlp.u_min)
n_constraints(nlp::ReducedSpaceEvaluator) = length(nlp.g_min)

constraints_type(::ReducedSpaceEvaluator) = :inequality
has_hessian(nlp::ReducedSpaceEvaluator) = nlp.has_hessian
number_batches_hessian(nlp::ReducedSpaceEvaluator) = nlp.has_hessian ? n_batches(nlp.hesslag) : 0

adjoint_jacobian(nlp::ReducedSpaceEvaluator, ::State) = nlp.state_jacobian.x.J
adjoint_jacobian(nlp::ReducedSpaceEvaluator, ::Control) = nlp.state_jacobian.u.J

# Getters
Base.get(nlp::ReducedSpaceEvaluator, ::Constraints) = nlp.constraints
function Base.get(nlp::ReducedSpaceEvaluator, ::State)
    x = similar(nlp.λ) ; fill!(x, 0)
    ExaPF.get!(nlp.model, State(), x, nlp.buffer)
    return x
end
Base.get(nlp::ReducedSpaceEvaluator, ::ExaPF.PhysicalState) = nlp.buffer

# Physics
Base.get(nlp::ReducedSpaceEvaluator, ::PS.VoltageMagnitude) = nlp.buffer.vmag
Base.get(nlp::ReducedSpaceEvaluator, ::PS.VoltageAngle) = nlp.buffer.vang
Base.get(nlp::ReducedSpaceEvaluator, ::PS.ActivePower) = nlp.buffer.pgen
Base.get(nlp::ReducedSpaceEvaluator, ::PS.ReactivePower) = nlp.buffer.qgen
function Base.get(nlp::ReducedSpaceEvaluator, attr::PS.AbstractNetworkAttribute)
    return ExaPF.get(nlp.model, attr)
end

# Setters
function setvalues!(nlp::ReducedSpaceEvaluator, attr::PS.AbstractNetworkValues, values)
    ExaPF.setvalues!(nlp.model, attr, values)
end
function setvalues!(nlp::ReducedSpaceEvaluator, attr::PS.ActiveLoad, values)
    ExaPF.setvalues!(nlp.buffer, attr, values)
end
function setvalues!(nlp::ReducedSpaceEvaluator, attr::PS.ReactiveLoad, values)
    ExaPF.setvalues!(nlp.buffer, attr, values)
end

# Transfer network values inside buffer
function transfer!(
    nlp::ReducedSpaceEvaluator, vm, va, pg, qg,
)
    setvalues!(nlp.buffer, PS.VoltageMagnitude(), vm)
    setvalues!(nlp.buffer, PS.VoltageAngle(), va)
    setvalues!(nlp.buffer, PS.ActivePower(), pg)
    setvalues!(nlp.buffer, PS.ReactivePower(), qg)
end

# Initial position
function initial(nlp::ReducedSpaceEvaluator)
    u = similar(nlp.u_min) ; fill!(u, 0.0)
    return ExaPF.get!(nlp.model, Control(), u, nlp.buffer)
end

# Bounds
bounds(nlp::ReducedSpaceEvaluator, ::Variables) = (nlp.u_min, nlp.u_max)
bounds(nlp::ReducedSpaceEvaluator, ::Constraints) = (nlp.g_min, nlp.g_max)

## Callbacks
function update!(nlp::ReducedSpaceEvaluator, u)
    jac_x = nlp.state_jacobian.x
    # Transfer control u into the network cache
    ExaPF.transfer!(nlp.model, nlp.buffer, u)
    # Get corresponding point on the manifold
    conv = ExaPF.powerflow(
        nlp.model,
        jac_x,
        nlp.buffer,
        nlp.powerflow_solver;
        linear_solver=nlp.linear_solver
    )

    if !conv.has_converged
        error("Newton-Raphson algorithm failed to converge ($(conv.norm_residuals))")
        return conv
    end

    # Evaluate Jacobian of power flow equation on current u
    AutoDiff.jacobian!(nlp.model, nlp.state_jacobian.u, nlp.buffer)
    # Specify that constraint's Jacobian is not up to date
    nlp.update_jacobian = nlp.has_jacobian
    # Update Hessian factorization
    if !isnothing(nlp.hesslag)
        ∇gₓ = nlp.state_jacobian.x.J
        update_factorization!(nlp.hesslag, ∇gₓ)
        # Update values for Hessian's AutoDiff
        ExaPF.update_hessian!(nlp.model, nlp.hesslag.hess, nlp.buffer)
    end
    return conv
end

# TODO: determine if we should include λ' * g(x, u), even if ≈ 0
function objective(nlp::ReducedSpaceEvaluator, u)
    # Take as input the current cache, updated previously in `update!`.
    return ExaPF.cost_production(nlp.model, nlp.buffer)
end

function constraint!(nlp::ReducedSpaceEvaluator, g, u)
    ϕ = nlp.buffer
    mf = 1::Int
    mt = 0::Int
    for cons in nlp.constraints
        m_ = ExaPF.size_constraint(nlp.model, cons)::Int
        mt += m_
        cons_ = @view(g[mf:mt])
        cons(nlp.model, cons_, ϕ)
        mf += m_
    end
end

###
# First-order code
####
#
# compute inplace reduced gradient (g = ∇fᵤ + (∇gᵤ')*λ)
# equivalent to: g = ∇fᵤ - (∇gᵤ')*λ_neg
# (take λₖ_neg to avoid computing an intermediate array)
function reduced_gradient!(
    nlp::ReducedSpaceEvaluator, grad, ∂fₓ, ∂fᵤ, λ, u,
)
    ∇gᵤ = nlp.state_jacobian.u.J
    ∇gₓ = nlp.state_jacobian.x.J

    # λ = ∇gₓ' \ ∂fₓ
    LS.rdiv!(nlp.backward_solver, λ, ∇gₓ, ∂fₓ)

    grad .= ∂fᵤ
    mul!(grad, transpose(∇gᵤ), λ, -1.0, 1.0)
    return
end
function reduced_gradient!(nlp::ReducedSpaceEvaluator, grad, ∂fₓ, ∂fᵤ, u)
    reduced_gradient!(nlp::ReducedSpaceEvaluator, grad, ∂fₓ, ∂fᵤ, nlp.λ, u)
end

# Compute only full gradient wrt x and u
function full_gradient!(nlp::ReducedSpaceEvaluator, gx, gu, u)
    buffer = nlp.buffer
    ∂obj = nlp.obj_stack
    # Evaluate adjoint of cost function and update inplace AdjointStackObjective
    ExaPF.gradient_objective!(nlp.model, ∂obj, buffer)
    copyto!(gx, ∂obj.stack.∇fₓ)
    copyto!(gu, ∂obj.stack.∇fᵤ)
end

function gradient!(nlp::ReducedSpaceEvaluator, g, u)
    buffer = nlp.buffer
    # Evaluate adjoint of cost function and update inplace AdjointStackObjective
    ExaPF.gradient_objective!(nlp.model, nlp.obj_stack, buffer)
    ∇fₓ, ∇fᵤ = nlp.obj_stack.stack.∇fₓ, nlp.obj_stack.stack.∇fᵤ

    reduced_gradient!(nlp, g, ∇fₓ, ∇fᵤ, u)
    return
end

function jacobian_structure(nlp::ReducedSpaceEvaluator)
    m, n = n_constraints(nlp), n_variables(nlp)
    nnzj = m * n
    rows = zeros(Int, nnzj)
    cols = zeros(Int, nnzj)
    jacobian_structure!(nlp, rows, cols)
    return rows, cols
end

function jacobian_structure!(nlp::ReducedSpaceEvaluator, rows, cols)
    m, n = n_constraints(nlp), n_variables(nlp)
    idx = 1
    for i in 1:n # number of variables
        for c in 1:m #number of constraints
            rows[idx] = c ; cols[idx] = i
            idx += 1
        end
    end
end

function _update_full_jacobian_constraints!(nlp)
    if nlp.update_jacobian
        ExaPF.update_full_jacobian!(nlp.model, nlp.constraint_jacobians, nlp.buffer)
        nlp.update_jacobian = false
    end
end

function jprod!(nlp::ReducedSpaceEvaluator, jm, u, v)
    nᵤ = length(u)
    m  = n_constraints(nlp)
    @assert nᵤ == size(v, 1)

    _update_full_jacobian_constraints!(nlp)
    H = nlp.hesslag
    ∇gᵤ = nlp.state_jacobian.u.J

    # Arrays
    Jx = nlp.constraint_jacobians.Jx
    Ju = nlp.constraint_jacobians.Ju
    z = H.z

    # init RHS
    mul!(z, ∇gᵤ, v)
    ldiv!(H.lu, z)

    # jv .= Ju * v .- Jx * z
    mul!(jm, Ju, v)
    mul!(jm, Jx, z, -1.0, 1.0)
    return
end

function full_jtprod!(nlp::ReducedSpaceEvaluator, jvx, jvu, u, v)
    fr_ = 0::Int
    for (cons, stack) in zip(nlp.constraints, nlp.cons_stacks)
        n = ExaPF.size_constraint(nlp.model, cons)::Int
        mask = fr_+1:fr_+n
        vv = @view v[mask]
        # Compute jtprod of current constraint
        ExaPF.jacobian_transpose_product!(nlp.model, stack, nlp.buffer, vv)
        jvx .+= stack.stack.∂x
        jvu .+= stack.stack.∂u
        fr_ += n
    end
end

function jtprod!(nlp::ReducedSpaceEvaluator, jv, u, v)
    ∂obj = nlp.obj_stack
    μ = nlp.buffer.balance
    jvx = ∂obj.stack.jvₓ ; fill!(jvx, 0)
    jvu = ∂obj.stack.jvᵤ ; fill!(jvu, 0)
    full_jtprod!(nlp, jvx, jvu, u, v)
    reduced_gradient!(nlp, jv, jvx, jvu, μ, u)
end

function ojtprod!(nlp::ReducedSpaceEvaluator, jv, u, σ, v)
    ∂obj = nlp.obj_stack
    jvx = ∂obj.stack.jvₓ ; fill!(jvx, 0)
    jvu = ∂obj.stack.jvᵤ ; fill!(jvu, 0)
    # compute gradient of objective
    full_gradient!(nlp, jvx, jvu, u)
    jvx .*= σ
    jvu .*= σ
    # compute transpose Jacobian vector product of constraints
    full_jtprod!(nlp, jvx, jvu, u, v)
    # Evaluate gradient in reduced space
    reduced_gradient!(nlp, jv, jvx, jvu, u)
    return
end

###
# Second-order code
####
# Single version
function full_hessprod!(nlp::ReducedSpaceEvaluator, hv::AbstractVector, y::AbstractVector, tgt::AbstractVector)
    nx, nu = ExaPF.get(nlp.model, ExaPF.NumberOfState()), ExaPF.get(nlp.model, ExaPF.NumberOfControl())
    H = nlp.hesslag
    AutoDiff.adj_hessian_prod!(nlp.model, H.hess, hv, nlp.buffer, y, tgt)
    ∂fₓ = @view hv[1:nx]
    ∂fᵤ = @view hv[nx+1:nx+nu]
    return ∂fₓ , ∂fᵤ
end

# Batch version
function full_hessprod!(nlp::ReducedSpaceEvaluator, hv::AbstractMatrix, y::AbstractMatrix, tgt::AbstractMatrix)
    nx, nu = ExaPF.get(nlp.model, ExaPF.NumberOfState()), ExaPF.get(nlp.model, ExaPF.NumberOfControl())
    H = nlp.hesslag
    ExaPF.batch_adj_hessian_prod!(nlp.model, H.hess, hv, nlp.buffer, y, tgt)
    # TODO: remove allocations
    ∂fₓ = hv[1:nx, :]
    ∂fᵤ = hv[nx+1:nx+nu, :]
    return ∂fₓ , ∂fᵤ
end

function hessprod!(nlp::ReducedSpaceEvaluator, hessvec, u, w)
    @assert nlp.hesslag != nothing

    nx = ExaPF.get(nlp.model, ExaPF.NumberOfState())
    nu = ExaPF.get(nlp.model, ExaPF.NumberOfControl())
    # TODO: remove
    nbus = ExaPF.get(nlp.model, ExaPF.PowerSystem.NumberOfBuses())
    H = nlp.hesslag
    ∇gᵤ = nlp.state_jacobian.u.J

    # Number of batches
    nbatch = size(w, 2)
    @assert nbatch == size(H.z, 2) == size(hessvec, 2)

    # Load variables and buffers
    tgt = H.tmp_tgt
    hv = H.tmp_hv
    y = H.y
    z = H.z
    ψ = H.ψ

    # Step 1: computation of first second-order adjoint
    mul!(z, ∇gᵤ, w, -1.0, 0.0)
    LinearAlgebra.ldiv!(H.lu, z)

    # Init tangent with z and w
    for i in 1:nbatch
        mxu = 1 + (i-1)*(nx+nu)
        mx = 1 + (i-1)*nx
        mu = 1 + (i-1)*nu
        copyto!(tgt, mxu,    z, mx, nx)
        copyto!(tgt, mxu+nx, w, mu, nu)
    end

    # Init adjoint
    fill!(y, 0.0)
    y[2*nbus+1:2*nbus+1] .= 1.0         # / objective
    y[1:nx] .-= nlp.λ  # / power balance

    # STEP 2: AutoDiff
    ∂fₓ, ∂fᵤ = full_hessprod!(nlp, hv, y, tgt)

    # STEP 3: computation of second second-order adjoint
    copyto!(ψ, ∂fₓ)
    LinearAlgebra.ldiv!(H.adjlu, ψ)

    hessvec .= ∂fᵤ
    mul!(hessvec, transpose(∇gᵤ), ψ, -1.0, 1.0)

    return
end

function hessian_lagrangian_penalty_prod!(
    nlp::ReducedSpaceEvaluator, hessvec, u, y, σ, D, w,
)
    @assert nlp.hesslag != nothing

    nbatch = size(w, 2)
    nx = ExaPF.get(nlp.model, ExaPF.NumberOfState())
    nu = ExaPF.get(nlp.model, ExaPF.NumberOfControl())
    # TODO: remove
    nbus = ExaPF.get(nlp.model, ExaPF.PowerSystem.NumberOfBuses())
    buffer = nlp.buffer
    H = nlp.hesslag
    ∇gᵤ = nlp.state_jacobian.u.J

    fill!(hessvec, 0.0)

    z = H.z
    ψ = H.ψ
    ∇gᵤ = nlp.state_jacobian.u.J
    mul!(z, ∇gᵤ, w, -1.0, 0.0)
    LinearAlgebra.ldiv!(H.lu, z)

    # Two vector products
    μ = H.y
    tgt = H.tmp_tgt
    hv = H.tmp_hv

    # Init tangent with z and w
    for i in 1:nbatch
        mxu = 1 + (i-1)*(nx+nu)
        mx = 1 + (i-1)*nx
        mu = 1 + (i-1)*nu
        copyto!(tgt, mxu,    z, mx, nx)
        copyto!(tgt, mxu+nx, w, mu, nu)
    end

    ## OBJECTIVE HESSIAN
    fill!(μ, 0.0)
    μ[1:nx] .-= nlp.λ  # / power balance
    μ[2*nbus+1:2*nbus+1] .= σ         # / objective
    # / constraints
    shift_m = nx
    shift_y = ExaPF.size_constraint(nlp.model, ExaPF.voltage_magnitude_constraints)
    for cons in [ExaPF.active_power_constraints, ExaPF.reactive_power_constraints]
        m = ExaPF.size_constraint(nlp.model, cons)::Int
        μ[shift_m+1:m+shift_m] .= view(y, shift_y+1:shift_y+m)
        shift_m += m
        shift_y += m
    end
    if nlp.constraints[end] == ExaPF.flow_constraints
        m = ExaPF.size_constraint(nlp.model, ExaPF.flow_constraints)::Int
        μ[2*nbus+2:2*nbus+1+m] .= view(y, shift_y+1:shift_y+m)
    end

    ∇²Lx, ∇²Lu = full_hessprod!(nlp, hv, μ, tgt)

    # Add Hessian of quadratic penalty
    m = length(y)
    diagjac = (nbatch > 1) ? similar(y, m, nbatch) : similar(y)
    _update_full_jacobian_constraints!(nlp)
    Jx = nlp.constraint_jacobians.Jx
    Ju = nlp.constraint_jacobians.Ju
    # ∇²Lx .+= Jx' * (D * (Jx * z)) .+ Jx' * (D * (Ju * w))
    # ∇²Lu .+= Ju' * (D * (Jx * z)) .+ Ju' * (D * (Ju * w))
    mul!(diagjac, Jx, z)
    mul!(diagjac, Ju, w, 1.0, 1.0)
    diagjac .*= D
    mul!(∇²Lx, Jx', diagjac, 1.0, 1.0)
    mul!(∇²Lu, Ju', diagjac, 1.0, 1.0)

    # Second order adjoint
    copyto!(ψ, ∇²Lx)
    LinearAlgebra.ldiv!(H.adjlu, ψ)

    hessvec .+= ∇²Lu
    mul!(hessvec, transpose(∇gᵤ), ψ, -1.0, 1.0)

    return
end

# Batch Hessian
macro define_batch_hessian(function_name, target_function, args...)
    fname_dispatch = Symbol("_" * String(function_name))
    fname = Symbol(function_name)
    argstup = Tuple(args)
    quote
        function $(esc(fname_dispatch))(nlp::ReducedSpaceEvaluator, hesslag::BatchHessianLagrangian, dest, $(map(esc, argstup)...))
            @assert has_hessian(nlp)
            @assert n_batches(hesslag) > 1
            n = n_variables(nlp)
            ∇²f = hesslag.hess
            nbatch = size(hesslag.tmp_hv, 2)

            # Allocate memory for tangents
            v = hesslag.tangents

            N = div(n, nbatch, RoundDown)
            for i in 1:N
                # Init tangents on CPU
                offset = (i-1) * nbatch
                set_batch_tangents!(v, offset, n, nbatch)
                # Contiguous views!
                hm = @view dest[:, nbatch * (i-1) + 1: nbatch * i]
                $target_function(nlp, hm, $(map(esc, argstup)...), v)
            end

            # Last slice
            last_batch = n - N*nbatch
            if last_batch > 0
                offset = n - nbatch
                set_batch_tangents!(v, offset, n, nbatch)

                hm = @view dest[:, (n - nbatch + 1) : n]
                $target_function(nlp, hm, $(map(esc, argstup)...), v)
            end
        end
        function $(esc(fname_dispatch))(nlp::ReducedSpaceEvaluator, hesslag::HessianLagrangian, dest, $(map(esc, argstup)...))
            @assert has_hessian(nlp)
            n = n_variables(nlp)
            v_cpu = zeros(n)
            v = similar(x)
            @inbounds for i in 1:n
                hv = @view dest[:, i]
                fill!(v_cpu, 0)
                v_cpu[i] = 1.0
                copyto!(v, v_cpu)
                $target_function(nlp, hv, $(map(esc, argstup)...), v)
            end
        end
        $(esc(fname))(nlp::ReducedSpaceEvaluator, dest, $(map(esc, argstup)...)) = $(esc(fname_dispatch))(nlp, nlp.hesslag, dest, $(map(esc, argstup)...))
    end
end

@define_batch_hessian hessian! hessprod! x
@define_batch_hessian hessian_lagrangian_penalty! hessian_lagrangian_penalty_prod! x y σ D
@define_batch_hessian jacobian! jprod! x


# Return lower-triangular matrix
function hessian_structure(nlp::ReducedSpaceEvaluator)
    n = n_variables(nlp)
    rows = Int[r for r in 1:n for c in 1:r]
    cols = Int[c for r in 1:n for c in 1:r]
    return rows, cols
end

# Utils function
function primal_infeasibility!(nlp::ReducedSpaceEvaluator, cons, u)
    constraint!(nlp, cons, u) # Evaluate constraints
    (n_inf, err_inf, n_sup, err_sup) = _check(cons, nlp.g_min, nlp.g_max)
    return max(err_inf, err_sup)
end
function primal_infeasibility(nlp::ReducedSpaceEvaluator, u)
    cons = similar(nlp.g_min) ; fill!(cons, 0)
    return primal_infeasibility!(nlp, cons, u)
end

# Printing
function sanity_check(nlp::ReducedSpaceEvaluator, u, cons)
    println("Check violation of constraints")
    print("Control  \t")
    (n_inf, err_inf, n_sup, err_sup) = _check(u, nlp.u_min, nlp.u_max)
    @printf("UB: %.4e (%d)    LB: %.4e (%d)\n",
            err_sup, n_sup, err_inf, n_inf)
    print("Constraints\t")
    (n_inf, err_inf, n_sup, err_sup) = _check(cons, nlp.g_min, nlp.g_max)
    @printf("UB: %.4e (%d)    LB: %.4e (%d)\n",
            err_sup, n_sup, err_inf, n_inf)
end

function Base.show(io::IO, nlp::ReducedSpaceEvaluator)
    n = n_variables(nlp)
    m = n_constraints(nlp)
    println(io, "A ReducedSpaceEvaluator object")
    println(io, "    * device: ", nlp.model.device)
    println(io, "    * #vars: ", n)
    println(io, "    * #cons: ", m)
    println(io, "    * constraints:")
    for cons in nlp.constraints
        println(io, "        - ", cons)
    end
    print(io, "    * linear solver: ", typeof(nlp.linear_solver))
end

function reset!(nlp::ReducedSpaceEvaluator)
    # Reset adjoint
    fill!(nlp.λ, 0)
    # Reset buffer
    ExaPF.init_buffer!(nlp.model, nlp.buffer)
    return
end

