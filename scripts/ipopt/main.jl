

using Printf
using Revise
using ExaPF
using Argos
using Ipopt
using SuiteSparse
using DelimitedFiles

using MathOptInterface
const MOI = MathOptInterface

include(joinpath(dirname(@__FILE__), "..", "common.jl"))
include(joinpath(dirname(@__FILE__), "..", "problems.jl"))

results = Dict()
function ipopt_optimizer()
    solver = Ipopt.Optimizer()
    # Default parameters
    MOI.set(solver, MOI.RawParameter("print_level"), 0)
    MOI.set(solver, MOI.RawParameter("tol"), 1e-4)
    attach_callback!(solver, aug, results)
    return solver
end

function dump_results(results)
    for k in [:feas, :voltage, :slack_power, :reactive_power, :line_constraints, :iter_count]
        dump_name = "auglag_$k.txt"
        writedlm(dump_name, results[k])
    end
end

function ipopt_subproblem(aug)
    optimizer = ipopt_optimizer()
    MOI.set(solver, MOI.RawParameter("print_level"), 5)
    solution = @time Argos.optimize!(optimizer, aug)
    MOI.empty!(optimizer)
end

function feasibility(x, xl, xu)
    feas = 0.0
    for i in eachindex(x)
        feas = max(feas, xl[i] - x[i])
        feas = max(feas, x[i] - xu[i])
    end
    return feas
end

function attach_callback!(solver::Ipopt.Optimizer, aug, results)
    inner = aug.inner
    model = Argos.backend(aug)
    buffer = get(aug, ExaPF.PhysicalState())

    m = Argos.n_constraints(inner)
    nv = ExaPF.size_constraint(model, ExaPF.voltage_magnitude_constraints)
    np = ExaPF.size_constraint(model, ExaPF.active_power_constraints)
    nq = ExaPF.size_constraint(model, ExaPF.reactive_power_constraints)
    nl = ExaPF.size_constraint(model, ExaPF.flow_constraints)

    g_L, g_U = Argos.bounds(inner, Argos.Constraints())
    v_lb, v_ub = ExaPF.bounds(model, ExaPF.voltage_magnitude_constraints)
    p_lb, p_ub = ExaPF.bounds(model, ExaPF.active_power_constraints)
    q_lb, q_ub = ExaPF.bounds(model, ExaPF.reactive_power_constraints)
    l_lb, l_ub = ExaPF.bounds(model, ExaPF.flow_constraints)

    c = zeros(m)
    cv = zeros(nv)
    cp = zeros(np)
    cq = zeros(nq)
    cl = zeros(nl)

    for k in [:feas, :voltage, :slack_power, :reactive_power, :line_constraints]
        if !haskey(results, k)
            results[k] = Float64[]
        end
    end
    if !haskey(results, :iter_count)
        results[:iter_count] = Int[]
    end

    function my_callback(
        prob::IpoptProblem,
        alg_mod::Cint,
        iter_count::Cint,
        obj_value::Float64,
        inf_pr::Float64,
        inf_du::Float64,
        mu::Float64,
        d_norm::Float64,
        regularization_size::Float64,
        alpha_du::Float64,
        alpha_pr::Float64,
        ls_trials::Cint,
    )
        Argos.constraint!(inner, c, prob.x)
        ExaPF.voltage_magnitude_constraints(model, cv, buffer)
        ExaPF.active_power_constraints(model, cp, buffer)
        ExaPF.reactive_power_constraints(model, cq, buffer)
        ExaPF.flow_constraints(model, cl, buffer)
        push!(results[:voltage],        feasibility(cv, v_lb, v_ub))
        push!(results[:slack_power],    feasibility(cp, p_lb, p_ub))
        push!(results[:reactive_power], feasibility(cq, q_lb, q_ub))
        push!(results[:line_constraints],   feasibility(cl, l_lb, l_ub))
        push!(results[:feas],   norm(aug.cons, Inf))
        push!(results[:iter_count], iter_count)
        return true
    end

    MOI.set(solver, Ipopt.CallbackFunction(), my_callback)
    return
end

# Augmented Lagrangian method
function solve_auglag(aug; max_iter=20, penalty=0.1, rate=10.0)
    # Init a penalty evaluator with initial penalty c₀
    algo = Argos.AugLagSolver(;
        max_iter=max_iter,
        max_inner_iter=100,
        α0=1.0,
        ρ0=penalty,
        rate=rate,
        ωtol=1e-3,
        verbose=1,
        inner_algo=:MOI,
        ε_dual=1e-5,
        ε_primal=1e-5,
    )
    aug.ρ = penalty
    x0 = Argos.initial(aug)

    solution = @time Argos.optimize!(algo, aug, x0; moi_optimizer=ipopt_optimizer)
    return aug, solution
end

datafile = "../../pglib-opf/pglib_opf_case118_ieee.m"
# datafile = PROBLEMS["case1354"]
aug = build_problem(datafile; scale=true)
solution = @time solve_auglag(aug; penalty=10.0, max_iter=5, rate=10.0)

