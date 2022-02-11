
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
mutable struct ReducedSpaceEvaluator{T, VI, VT, MT, Jacx, Jacu, JacCons, Hess, MyReduction} <: AbstractNLPEvaluator
    model::ExaPF.PolarForm{T, VI, VT, MT}
    nx::Int
    nu::Int
    mapx::VI
    mapu::VI
    mapxu::VI

    # Expressions
    basis::AutoDiff.AbstractExpression
    costs::AutoDiff.AbstractExpression
    constraints::AutoDiff.AbstractExpression

    # Buffers
    obj::T
    cons::VT
    grad::VT
    multipliers::VT
    wu::VT
    wx::VT
    λ::VT # adjoint of objective
    μ::VT # adjoint of Lagrangian

    u_min::VT
    u_max::VT
    g_min::VT
    g_max::VT

    # Stack
    stack::ExaPF.NetworkStack
    ∂stack::ExaPF.NetworkStack
    # Jacobians for Power flow
    Gx::Jacx
    Gu::Jacu
    # Jacobian for remaining constraints
    jac::JacCons

    hess::Hess
    reduction::MyReduction

    # Options
    linear_solver::LS.AbstractLinearSolver
    powerflow_solver::ExaPF.AbstractNonLinearSolver
    pf_buffer::ExaPF.NLBuffer
    is_jacobian_updated::Bool
    is_hessian_objective_updated::Bool
    is_hessian_lagrangian_updated::Bool
    is_adjoint_objective_updated::Bool
    is_adjoint_lagrangian_updated::Bool
    etc::Dict{Symbol, Any}
end

function ReducedSpaceEvaluator(
    model::PolarForm{T, VI, VT, MT};
    line_constraints=true,
    linear_solver=nothing,
    backward_solver=nothing,
    powerflow_solver=NewtonRaphson(tol=1e-10),
    want_jacobian=true,
    nbatch_hessian=1,
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
    powerflow = ExaPF.PowerFlowBalance(model)
    constraints_expr = [
        ExaPF.VoltageMagnitudePQ(model),
        ExaPF.PowerGenerationBounds(model),
    ]
    if line_constraints
        push!(constraints_expr, ExaPF.LineFlows(model))
    end
    constraints = ExaPF.MultiExpressions(constraints_expr)
    m = length(constraints)

    # Stacks
    stack = ExaPF.NetworkStack(model)
    ∂stack = ExaPF.NetworkStack(model)
    # Buffers
    obj = Inf
    cons = VT(undef, m)
    wx = VT(undef, nx)
    wu = VT(undef, nu)
    grad = VT(undef, nx+nu)
    y = VT(undef, m + nx + 1)
    λ = VT(undef, nx)
    μ = VT(undef, nx)
    pf_buffer = ExaPF.NLBuffer{VT}(nx)

    s_min, s_max = ExaPF.bounds(model, stack)
    u_min, u_max = s_min[mapu], s_max[mapu]
    g_min, g_max = ExaPF.bounds(model, constraints)
    # Remove bounds below a given threshold
    g_max = min.(g_max, 1e5)

    # Jacobians
    Gx = ExaPF.Jacobian(model, powerflow ∘ basis, mapx)
    Gu = ExaPF.Jacobian(model, powerflow ∘ basis, mapu)
    jac = ExaPF.Jacobian(model, constraints ∘ basis, mapxu)
    # Hessian of Lagrangian
    lagrangian_expr = [costs; powerflow; constraints_expr]
    lagrangian = ExaPF.MultiExpressions(lagrangian_expr)
    hess = ExaPF.FullHessian(model, lagrangian ∘ basis, mapxu)

    # Build Linear Algebra
    J = Gx.J
    _linear_solver = isnothing(linear_solver) ? LS.DirectSolver(J; nbatch=nbatch_hessian) : linear_solver
    S = ImplicitSensitivity(_linear_solver.factorization, Gu.J)
    redop = if nbatch_hessian > 1
        BatchReduction(model, S, nbatch_hessian)
    else
        Reduction(model, S)
    end

    etc = Dict{Symbol, Any}()

    return ReducedSpaceEvaluator{T,VI,VT,MT,typeof(Gx),typeof(Gu),typeof(jac),typeof(hess),typeof(redop)}(
        model, nx, nu, mapx, mapu, mapxu,
        basis, costs, constraints,
        obj, cons, grad, y, wu, wx, λ, μ,
        u_min, u_max, g_min, g_max,
        stack, ∂stack,
        Gx, Gu, jac, hess, redop,
        _linear_solver,
        powerflow_solver, pf_buffer,
        false, false, false, false, false, etc,
    )
end
function ReducedSpaceEvaluator(datafile::String; device=ExaPF.CPU(), options...)
    return ReducedSpaceEvaluator(ExaPF.PolarForm(datafile, device); options...)
end

model(nlp::ReducedSpaceEvaluator) = nlp.model
backend(nlp::ReducedSpaceEvaluator) = nlp

n_variables(nlp::ReducedSpaceEvaluator) = nlp.nu
n_constraints(nlp::ReducedSpaceEvaluator) = length(nlp.g_min)

constraints_type(::ReducedSpaceEvaluator) = :inequality
has_hessian(nlp::ReducedSpaceEvaluator) = true
has_hessian_lagrangian(nlp::ReducedSpaceEvaluator) = true

number_batches(nlp::ReducedSpaceEvaluator) = n_batches(nlp.reduction)

function get_hessian_buffer(nlp::ReducedSpaceEvaluator)
    if !haskey(nlp.etc,:hess)
        n = n_variables(nlp)
        nlp.etc[:hess] = zeros(n, n)
    end
    return nlp.etc[:hess]
end

# Getters
Base.get(nlp::ReducedSpaceEvaluator, ::Constraints) = nlp.constraints
function Base.get(nlp::ReducedSpaceEvaluator, ::State)
    return nlp.stack.input[nlp.mapx]
end

# Physics
Base.get(nlp::ReducedSpaceEvaluator, ::PS.VoltageMagnitude) = nlp.stack.vmag
Base.get(nlp::ReducedSpaceEvaluator, ::PS.VoltageAngle) = nlp.stack.vang
Base.get(nlp::ReducedSpaceEvaluator, ::PS.ActivePower) = nlp.stack.pgen
Base.get(nlp::ReducedSpaceEvaluator, ::PS.ReactivePower) = nlp.stack.qgen
function Base.get(nlp::ReducedSpaceEvaluator, attr::PS.AbstractNetworkAttribute)
    return ExaPF.get(nlp.model, attr)
end
get_nnzh(nlp::ReducedSpaceEvaluator) = n_variables(nlp)^2

# Setters
function setvalues!(nlp::ReducedSpaceEvaluator, attr::PS.AbstractNetworkValues, values)
    ExaPF.setvalues!(nlp.model, attr, values)
end
function setvalues!(nlp::ReducedSpaceEvaluator, attr::PS.ActiveLoad, values)
    ExaPF.setvalues!(nlp.stack, attr, values)
end
function setvalues!(nlp::ReducedSpaceEvaluator, attr::PS.ReactiveLoad, values)
    ExaPF.setvalues!(nlp.stack, attr, values)
end

# Transfer network values inside buffer
# function transfer!(
#     nlp::ReducedSpaceEvaluator, vm, va, pg, qg,
# )

#     setvalues!(nlp.buffer, PS.VoltageMagnitude(), vm)
#     setvalues!(nlp.buffer, PS.VoltageAngle(), va)
#     setvalues!(nlp.buffer, PS.ActivePower(), pg)
#     setvalues!(nlp.buffer, PS.ReactivePower(), qg)
# end

# Initial position
function initial(nlp::ReducedSpaceEvaluator{T, VI, VT, MT}) where {T, VI, VT, MT}
    u = VT(undef, nlp.nu)
    copyto!(u, nlp.stack, nlp.mapu)
    return u
end

# Bounds
bounds(nlp::ReducedSpaceEvaluator, ::Variables) = (nlp.u_min, nlp.u_max)
bounds(nlp::ReducedSpaceEvaluator, ::Constraints) = (nlp.g_min, nlp.g_max)

## Callbacks
function update!(nlp::ReducedSpaceEvaluator, u)
    # Transfer control u into the network cache
    copyto!(nlp.stack, nlp.mapu, u)

    # Get corresponding point on the manifold
    conv = ExaPF.nlsolve!(
        nlp.powerflow_solver,
        nlp.Gx,
        nlp.stack;
        linear_solver=nlp.linear_solver,
        nl_buffer=nlp.pf_buffer,
    )
    if !conv.has_converged
        println("Newton-Raphson algorithm failed to converge ($(conv.norm_residuals))")
        return conv
    end

    # Full forward pass
    nlp.basis(nlp.stack.ψ, nlp.stack)
    nlp.obj = sum(nlp.costs(nlp.stack))
    nlp.constraints(nlp.cons, nlp.stack)

    # Evaluate Jacobian of power flow equation on current u
    ExaPF.jacobian!(nlp.Gu, nlp.stack)

    # Update reduction
    update!(nlp.reduction)

    nlp.is_jacobian_updated = false
    nlp.is_adjoint_lagrangian_updated = false
    nlp.is_adjoint_objective_updated = false
    nlp.is_hessian_objective_updated = false
    nlp.is_hessian_lagrangian_updated = false

    return conv
end

objective(nlp::ReducedSpaceEvaluator, u) = nlp.obj
constraint!(nlp::ReducedSpaceEvaluator, cons, u) = copyto!(cons, nlp.cons)

###
# First-order code
####

# compute inplace reduced gradient (g = ∇fᵤ + (∇gᵤ')*λ)
# equivalent to: g = ∇fᵤ - (∇gᵤ')*λ_neg
# (take λₖ_neg to avoid computing an intermediate array)
function _adjoint_solve!(
    nlp::ReducedSpaceEvaluator, grad, ∇f, u, λ,
)
    nx, nu = nlp.nx, nlp.nu
    lujac = nlp.linear_solver.factorization
    Gu = nlp.Gu.J
    Gx = nlp.Gx.J

    ∇fx = @view ∇f[1:nx]
    ∇fu = @view ∇f[1+nx:nx+nu]

    # λ = ∇gₓ' \ ∂fₓ
    LinearAlgebra.ldiv!(λ, lujac', ∇fx)

    grad .= ∇fu
    mul!(grad, transpose(Gu), λ, -1.0, 1.0)
    return
end

function gradient!(nlp::ReducedSpaceEvaluator, grad, u)
    ∇f = nlp.grad
    ExaPF.empty!(nlp.∂stack)
    ExaPF.adjoint!(nlp.costs, nlp.∂stack, nlp.stack, 1.0)
    ExaPF.adjoint!(nlp.basis, nlp.∂stack, nlp.stack, nlp.∂stack.ψ)
    copyto!(∇f, nlp.∂stack, nlp.mapxu)
    _adjoint_solve!(nlp, grad, ∇f, u, nlp.λ)
    nlp.is_adjoint_objective_updated = true
    return
end

function jprod!(nlp::ReducedSpaceEvaluator, jm, u, v)
    if !nlp.is_jacobian_updated
        ExaPF.jacobian!(nlp.jac, nlp.stack)
        nlp.is_jacobian_updated = true
    end

    J = nlp.jac.J
    Gu = nlp.Gu.J
    direct_reduction!(nlp.reduction, jm, J, Gu, v)
    return
end

function jtprod!(nlp::ReducedSpaceEvaluator, jv, u, v)
    grad = nlp.grad
    empty!(nlp.∂stack)
    ExaPF.adjoint!(nlp.constraints, nlp.∂stack, nlp.stack, v)
    ExaPF.adjoint!(nlp.basis, nlp.∂stack, nlp.stack, nlp.∂stack.ψ)
    copyto!(grad, nlp.∂stack, nlp.mapxu)
    μ = nlp.wx
    _adjoint_solve!(nlp, jv, grad, u, μ)
end

function ojtprod!(nlp::ReducedSpaceEvaluator, jv, u, σ, v)
    grad = nlp.grad
    ExaPF.empty!(nlp.∂stack)
    # Accumulate adjoint / objective
    ExaPF.adjoint!(nlp.costs, nlp.∂stack, nlp.stack, σ)
    # Accumulate adjoint / constraints
    ExaPF.adjoint!(nlp.constraints, nlp.∂stack, nlp.stack, v)
    # Accumulate adjoint / basis
    ExaPF.adjoint!(nlp.basis, nlp.∂stack, nlp.stack, nlp.∂stack.ψ)
    copyto!(grad, nlp.∂stack, nlp.mapxu)
    # Evaluate gradient in reduced space
    _adjoint_solve!(nlp, jv, grad, u, nlp.μ)
    nlp.is_adjoint_lagrangian_updated = true
    return
end

###
# Second-order code
####
# Single version
function update_full_hessian!(nlp::ReducedSpaceEvaluator, y::AbstractVector)
    if !nlp.is_hessian_updated
        nlp.is_hessian_updated = true
    end
end
function hessprod!(nlp::ReducedSpaceEvaluator, hessvec, u, w)
    nx, nu = nlp.nx, nlp.nu
    m = length(nlp.constraints)
    # Check that the first-order adjoint is correct
    if !nlp.is_adjoint_objective_updated
        g = nlp.wu
        gradient!(nlp, g, u)
    end
    y = nlp.multipliers
    # Init adjoint
    fill!(y, 0.0)
    y[1] = 1.0         # / objective
    y[2:nx+1] .-= nlp.λ       # / power balance
    # Update Hessian
    if !nlp.is_hessian_objective_updated
        ExaPF.hessian!(nlp.hess, nlp.stack, y)
        nlp.is_hessian_objective_updated = true
        nlp.is_hessian_lagrangian_updated = false
    end
    H = nlp.hess.H
    Gu = nlp.Gu.J
    adjoint_adjoint_reduction!(nlp.reduction, hessvec, H, w)
end

function hessian_lagrangian_prod!(
    nlp::ReducedSpaceEvaluator, hessvec, u, μ, σ, w,
)
    nx, nu = nlp.nx, nlp.nu
    m = length(nlp.constraints)
    if !nlp.is_adjoint_lagrangian_updated
        g = nlp.wu
        ojtprod!(nlp, g, u, σ, μ)
    end
    y = nlp.multipliers
    # Init adjoint
    fill!(y, 0.0)
    y[1:1] .= σ           # / objective
    y[2:nx+1] .-= nlp.μ       # / power balance
    y[nx+2:nx+1+m] .= μ       # / constraints
    # Update Hessian
    if !nlp.is_hessian_lagrangian_updated
        ExaPF.hessian!(nlp.hess, nlp.stack, y)
        nlp.is_hessian_lagrangian_updated = true
        nlp.is_hessian_objective_updated = false
    end
    # Get sparse matrices
    H = nlp.hess.H
    Gu = nlp.Gu.J
    adjoint_adjoint_reduction!(nlp.reduction, hessvec, H, w)
end

function hessian_lagrangian_penalty_prod!(
    nlp::ReducedSpaceEvaluator, hessvec, u, λ, σ, D, w,
)
    nx, nu = nlp.nx, nlp.nu
    m = length(nlp.constraints)
    if true #!nlp.is_adjoint_lagrangian_updated
        g = nlp.wu
        ojtprod!(nlp, g, u, σ, λ)
    end
    y = nlp.multipliers
    # Init adjoint
    fill!(y, 0.0)
    y[1:1] .= σ           # / objective
    y[2:nx+1] .-= nlp.μ       # / power balance
    y[nx+2:nx+1+m] .= λ       # / constraints
    # Update Hessian
    if true #!nlp.is_hessian_lagrangian_updated
        ExaPF.hessian!(nlp.hess, nlp.stack, y)
        ExaPF.jacobian!(nlp.jac, nlp.stack)
        nlp.is_hessian_lagrangian_updated = true
        nlp.is_hessian_objective_updated = false
    end
    # Get sparse matrices
    H = nlp.hess.H
    Gu = nlp.Gu.J
    J = nlp.jac.J
    W = H + J' * Diagonal(D) * J

    adjoint_adjoint_reduction!(nlp.reduction, hessvec, W, w)
end

# Batch Hessian
macro define_batch_callback(function_name, target_function, args...)
    fname_dispatch = Symbol("_" * String(function_name))
    fname = Symbol(function_name)
    argstup = Tuple(args)
    quote
        function $(esc(fname_dispatch))(nlp::ReducedSpaceEvaluator, reduction::BatchReduction, dest, $(map(esc, argstup)...))
            @assert has_hessian(nlp)
            @assert n_batches(reduction) > 1
            n = n_variables(nlp)
            nbatch = n_batches(reduction)

            # Allocate memory for tangents
            v = reduction.tangents

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
        function $(esc(fname_dispatch))(nlp::ReducedSpaceEvaluator, reduction::Reduction, dest, $(map(esc, argstup)...))
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
        $(esc(fname))(nlp::ReducedSpaceEvaluator, dest, $(map(esc, argstup)...)) = $(esc(fname_dispatch))(nlp, nlp.reduction, dest, $(map(esc, argstup)...))
    end
end

@define_batch_callback hessian! hessprod! x
@define_batch_callback hessian_lagrangian! hessian_lagrangian_prod! x y σ
@define_batch_callback hessian_lagrangian_penalty! hessian_lagrangian_penalty_prod! x y σ D
@define_batch_callback jacobian! jprod! x

function Base.show(io::IO, nlp::ReducedSpaceEvaluator)
    n = n_variables(nlp)
    m = n_constraints(nlp)
    println(io, "A ReducedSpaceEvaluator object")
    println(io, "    * device: ", nlp.model.device)
    println(io, "    * #vars: ", n)
    println(io, "    * #cons: ", m)
    println(io, "    * constraints:")
    # for cons in nlp.constraints
    #     println(io, "        - ", cons)
    # end
    print(io, "    * linear solver: ", typeof(nlp.linear_solver))
end

function reset!(nlp::ReducedSpaceEvaluator)
    # Reset adjoint
    fill!(nlp.λ, 0)
    fill!(nlp.μ, 0)
    fill!(nlp.multipliers, 0)
    fill!(nlp.grad, 0)
    empty!(nlp.stack)
    empty!(nlp.∂stack)
    ExaPF.init!(nlp.model, nlp.stack)
    nlp.is_jacobian_updated = false
    nlp.is_hessian_lagrangian_updated = false
    nlp.is_hessian_objective_updated = false
    nlp.is_adjoint_lagrangian_updated = false
    nlp.is_adjoint_objective_updated = false
    return
end

