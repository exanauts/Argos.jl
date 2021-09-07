using DelimitedFiles

function check_feasibility(aug)
    inner = aug.inner
    model = ExaOpt.backend(aug)
    buffer = get(aug, ExaPF.PhysicalState())

    m = ExaOpt.n_constraints(inner)
    nv = ExaPF.size_constraint(model, ExaPF.voltage_magnitude_constraints)
    np = ExaPF.size_constraint(model, ExaPF.active_power_constraints)
    nq = ExaPF.size_constraint(model, ExaPF.reactive_power_constraints)
    nl = ExaPF.size_constraint(model, ExaPF.flow_constraints)

    g_L, g_U  = ExaOpt.bounds(inner, ExaOpt.Constraints())
    v_lb, v_ub = ExaPF.bounds(model, ExaPF.voltage_magnitude_constraints)
    p_lb, p_ub = ExaPF.bounds(model, ExaPF.active_power_constraints)
    q_lb, q_ub = ExaPF.bounds(model, ExaPF.reactive_power_constraints)
    l_lb, l_ub = ExaPF.bounds(model, ExaPF.flow_constraints)

    c = zeros(m)
    cv = zeros(nv)
    cp = zeros(np)
    cq = zeros(nq)
    cl = zeros(nl)

    results = Dict()
    ExaPF.voltage_magnitude_constraints(model, cv, buffer)
    ExaPF.active_power_constraints(model, cp, buffer)
    ExaPF.reactive_power_constraints(model, cq, buffer)
    ExaPF.flow_constraints(model, cl, buffer)
    results[:voltage] =        ExaOpt.max_infeasibility_rel(cv, v_lb, v_ub)
    results[:slack_power] =    ExaOpt.max_infeasibility_rel(cp, p_lb, p_ub)
    results[:reactive_power] = ExaOpt.max_infeasibility_rel(cq, q_lb, q_ub)
    results[:line_constraints] =   ExaOpt.max_infeasibility_rel(cl, l_lb, l_ub)
    results[:feas] =   norm(aug.cons, Inf)
    return results
end

function dump_results(tracker::ExaOpt.NLPTracker, dirdump="")
    corr = [
        ("pr_feas", :primal_infeasibility),
        ("du_feas", :dual_infeasibility),
        ("pf_feas", :inf_power_flow),
        ("reactive_feas", :inf_reactive_power),
        ("active_power_feas", :inf_active_power),
        ("line_flow_feas", :inf_line_constraints),
        ("volt_feas", :inf_voltage),
    ]

    for (n, f) in corr
        dump_name = joinpath(dirdump, "$n.txt")
        writedlm(dump_name, getfield(tracker, f))
    end
    dump_name = joinpath(dirdump, "iter.txt")
    writedlm(dump_name, tracker.ext[:current_iter])
end
