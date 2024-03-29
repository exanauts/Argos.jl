# MOI wrapper

"""
    MOIEvaluator <: MOI.AbstractNLPEvaluator

Bridge from a [`AbstractNLPEvaluator`](@ref) to a `MOI.AbstractNLPEvaluator`.

## Example

```julia-repl
julia> datafile = "case9.m"  # specify a path to a MATPOWER instance

julia> nlp = Argos.ReducedSpaceEvaluator(datafile);

julia> ev = Argos.MOIEvaluator(nlp)

```

## Attributes

* `nlp::AbstractNLPEvaluator`: the underlying `ExaPF` problem.
* `hash_x::UInt`: hash of the last evaluated variable `x`
* `has_hess::Bool` (default: `false`): if `true`, pass a Hessian structure to MOI.

"""
mutable struct MOIEvaluator{Evaluator<:AbstractNLPEvaluator} <: MOI.AbstractNLPEvaluator
    nlp::Evaluator
    hash_x::UInt
    has_hess::Bool
    counters::NLPModels.Counters
    timers::NLPTimers
end
# MOI needs Hessian of Lagrangian function
function MOIEvaluator(nlp)
    return MOIEvaluator(nlp, UInt64(0), has_hessian_lagrangian(nlp), NLPModels.Counters(), NLPTimers())
end

function _update!(ev::MOIEvaluator, x)
    hx = hash(x)
    if hx != ev.hash_x
        ev.timers.update_time += @elapsed begin
            update!(ev.nlp, x)
        end
        ev.hash_x = hx
    end
end

MOI.features_available(ev::MOIEvaluator) = ev.has_hess ? [:Grad, :Hess] : [:Grad]
MOI.initialize(ev::MOIEvaluator, features) = nothing

function MOI.jacobian_structure(ev::MOIEvaluator)
    rows, cols = jacobian_structure(ev.nlp)
    return Tuple{Int, Int}[(r, c) for (r, c) in zip(rows, cols)]
end

function MOI.hessian_lagrangian_structure(ev::MOIEvaluator)
    n = n_variables(ev.nlp)
    rows, cols = hessian_structure(ev.nlp)
    return Tuple{Int, Int}[(r, c) for (r, c) in zip(rows, cols)]
end

function MOI.eval_objective(ev::MOIEvaluator, x)
    _update!(ev, x)
    ev.counters.neval_obj += 1
    ev.timers.obj_time += @elapsed begin
        obj = objective(ev.nlp, x)
    end
    return obj
end

function MOI.eval_objective_gradient(ev::MOIEvaluator, g, x)
    _update!(ev, x)
    ev.counters.neval_grad += 1
    ev.timers.grad_time += @elapsed begin
        gradient!(ev.nlp, g, x)
    end
end

function MOI.eval_constraint(ev::MOIEvaluator, cons, x)
    _update!(ev, x)
    ev.counters.neval_cons += 1
    ev.timers.cons_time += @elapsed begin
        constraint!(ev.nlp, cons, x)
    end
end

function MOI.eval_constraint_jacobian(ev::MOIEvaluator, jac, x)
    _update!(ev, x)
    ev.counters.neval_jac += 1
    ev.timers.jacobian_time += @elapsed begin
        jacobian_coo!(ev.nlp, jac, x)
    end
end

function MOI.eval_hessian_lagrangian(ev::MOIEvaluator, hess, x, σ, μ)
    _update!(ev, x)
    ev.counters.neval_hess += 1
    ev.timers.hessian_time += @elapsed begin
        hessian_lagrangian_coo!(ev.nlp, hess, x, μ, σ)
    end
end

function MOI.eval_hessian_lagrangian_product(ev::MOIEvaluator, hv, x, v, σ, μ)
    _update!(ev, x)
    ev.counters.neval_prod += 1
    ev.timers.hessprod_time += @elapsed begin
        hessian_lagrangian_prod!(ev.nlp, hv, x, μ, σ, v)
    end
end

function MOI.NLPBlockData(nlp::AbstractNLPEvaluator)
    lb, ub = bounds(nlp, Constraints())
    ev = MOIEvaluator(nlp)
    return MOI.NLPBlockData(MOI.NLPBoundsPair.(lb, ub), ev, true)
end

# MOI-based solver
function build!(optimizer::MOI.AbstractOptimizer, nlp::AbstractNLPEvaluator)
    block_data = MOI.NLPBlockData(nlp)
    u♭, u♯ = bounds(nlp, Variables())
    n = n_variables(nlp)
    u = MOI.add_variables(optimizer, n)
    # Set bounds
    MOI.add_constraints(
        optimizer, u, MOI.LessThan.(u♯),
    )
    MOI.add_constraints(
        optimizer, u, MOI.GreaterThan.(u♭),
    )
    MOI.set(optimizer, MOI.NLPBlock(), block_data)
    MOI.set(optimizer, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    return u
end

function optimize!(optimizer::MOI.AbstractOptimizer, nlp::AbstractNLPEvaluator, x0)
    u = build!(optimizer, nlp)
    MOI.set(optimizer, MOI.VariablePrimalStart(), u, x0)
    MOI.optimize!(optimizer)
    x_opt = MOI.get(optimizer, MOI.VariablePrimal(), u)
    y_opt = MOI.get(optimizer, MOI.NLPBlockDual())
    solution = (
        status=MOI.get(optimizer, MOI.TerminationStatus()),
        minimum=MOI.get(optimizer, MOI.ObjectiveValue()),
        minimizer=x_opt,
        dual=y_opt,
    )
    return solution
end
optimize!(optimizer, nlp::AbstractNLPEvaluator) = optimize!(optimizer, nlp, initial(nlp))

