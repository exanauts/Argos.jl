
using ExaPF
using JuMP
using Ipopt
using LinearAlgebra
const PS = ExaPF.PowerSystem


constraint_index(cons::Vector{NonlinearConstraintRef{ScalarShape}}) = getfield.(JuMP.index.(cons), :value)

function build_opf_model(polar, buffer, solver; line_constraints=true)
    nbus = ExaPF.get(polar, PS.NumberOfBuses())
    ngen = ExaPF.get(polar, PS.NumberOfGenerators())
    nlines = ExaPF.get(polar, PS.NumberOfLines())

    pf = polar.network
    baseMVA = pf.baseMVA

    pg_min, pg_max = PS.bounds(pf, PS.Generators(), PS.ActivePower())
    qg_min, qg_max = PS.bounds(pf, PS.Generators(), PS.ReactivePower())
    vm_min, vm_max = PS.bounds(pf, PS.Buses(), PS.VoltageMagnitude())

    flow_min, flow_max = PS.bounds(pf, PS.Lines(), PS.ActivePower())

    flow_max = min.(1e5, flow_max)

    vm0 = buffer.vmag
    va0 = buffer.vang
    pg0 = buffer.pgen
    qg0 = buffer.qgen

    # Pd = buffer.pload
    # Qd = buffer.qload

    cost_coefs = PS.get_costs_coefficients(pf)

    bus2gen = PS.get_bus_generators(pf.buses, pf.generators, pf.bus_to_indexes)

    # Power flow data
    Ybus = pf.Ybus
    rows = Ybus.rowval
    yvals = Ybus.nzval
    g_ij = real.(yvals)
    b_ij = imag.(yvals)

    # Topology
    top = polar.topology

    #=
        Build model
    =#

    opfmodel = Model(solver)

    # VARIABLES
    @variable(opfmodel, pg_min[i] <= Pg[i=1:ngen] <= pg_max[i], start=pg0[i])
    @variable(opfmodel, qg_min[i] <= Qg[i=1:ngen] <= qg_max[i], start=qg0[i])
    @variable(opfmodel, vm_min[i] <= Vm[i=1:nbus] <= vm_max[i], start=vm0[i])
    @variable(opfmodel, Va[i=1:nbus], start=va0[i])

    @variable(opfmodel, Pd[i=1:nbus])
    @variable(opfmodel, Qd[i=1:nbus])
    JuMP.fix.(Pd, buffer.pload)
    JuMP.fix.(Qd, buffer.qload)

    # Power-flow constraints
    ## active
    opfmodel.ext[:active_pf] = @NLconstraint(
        opfmodel, [b=1:nbus],
        Vm[b] * sum(
            Vm[rows[c]] * (g_ij[c] * cos(Va[b] - Va[rows[c]]) + b_ij[c] * sin(Va[b] - Va[rows[c]]))
            for c in (Ybus.colptr[b]):(Ybus.colptr[b+1]-1)
        ) == (sum(Pg[g] for g in get(bus2gen, b, Int[])) - Pd[b])
    )
    ## reactive
    opfmodel.ext[:reactive_pf] = @NLconstraint(
        opfmodel, [b=1:nbus],
        Vm[b] * sum(
            Vm[rows[c]] * (g_ij[c] * sin(Va[b] - Va[rows[c]]) - b_ij[c] * cos(Va[b] - Va[rows[c]]))
            for c in (Ybus.colptr[b]):(Ybus.colptr[b+1]-1)) == (sum(Qg[g] for g in get(bus2gen, b, Int[])) - Qd[b])
    )

    # Line constraints
    if line_constraints
        f = top.f_buses
        t = top.t_buses
        ## from lines
        yff_abs = top.yff_re.^2 .+ top.yff_im.^2
        yft_abs = top.yft_re.^2 .+ top.yft_im.^2
        yre_fr =   top.yff_re .* top.yft_re .+ top.yff_im .* top.yft_im
        yim_fr = - top.yff_re .* top.yft_im .+ top.yff_im .* top.yft_re

        opfmodel.ext[:line_fr] = @NLconstraint(
            opfmodel, [ℓ=1:nlines],
            Vm[f[ℓ]]^2 * (
                yff_abs[ℓ] * Vm[f[ℓ]]^2 + yft_abs[ℓ] * Vm[t[ℓ]]^2 +
                2.0 * Vm[f[ℓ]] * Vm[t[ℓ]] * (yre_fr[ℓ] * cos(Va[f[ℓ]]-Va[t[ℓ]]) - yim_fr[ℓ] * sin(Va[f[ℓ]]-Va[t[ℓ]]))
            ) <= flow_max[ℓ]
        )

        ## to lines
        ytf_abs = top.ytf_re.^2 .+ top.ytf_im.^2
        ytt_abs = top.ytt_re.^2 .+ top.ytt_im.^2
        yre_to =   top.ytf_re .* top.ytt_re .+ top.ytf_im .* top.ytt_im
        yim_to = - top.ytf_re .* top.ytt_im .+ top.ytf_im .* top.ytt_re

        opfmodel.ext[:line_to] = @NLconstraint(
            opfmodel, [ℓ=1:nlines],
            Vm[t[ℓ]]^2 * (
                ytf_abs[ℓ] * Vm[f[ℓ]]^2 + ytt_abs[ℓ] * Vm[t[ℓ]]^2 +
                2.0 * Vm[f[ℓ]] * Vm[t[ℓ]] * (yre_to[ℓ] * cos(Va[f[ℓ]]-Va[t[ℓ]]) - yim_to[ℓ] * sin(Va[f[ℓ]]-Va[t[ℓ]]))
            ) <= flow_max[ℓ]
        )
    end

    # Objective
    @objective(
        opfmodel,
        Min,
        sum(
            cost_coefs[g, 4] * Pg[g]^2 + cost_coefs[g, 3] * Pg[g] + cost_coefs[g, 2]
            for g in 1:ngen
        )
    )

    opfmodel.ext[:exapf] = pf

    return opfmodel
end

# Need a quick fix in Ipopt.jl to work properly!
function attach_callback!(opfmodel::Model)
    pf_active = constraint_index(opfmodel.ext[:active_pf])
    pf_reactive = constraint_index(opfmodel.ext[:reactive_pf])
    index_line_fr = constraint_index(opfmodel.ext[:line_fr])
    index_line_to = constraint_index(opfmodel.ext[:line_to])
    line_flow = [index_line_fr; index_line_to]

    opfmodel.ext[:feas_active_pf] = Float64[]
    opfmodel.ext[:feas_reactive_pf] = Float64[]
    opfmodel.ext[:feas_line_flow] = Float64[]

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
        push!(opfmodel.ext[:feas_active_pf],   norm(prob.g[pf_active], Inf))
        push!(opfmodel.ext[:feas_reactive_pf], norm(prob.g[pf_reactive], Inf))
        push!(opfmodel.ext[:feas_line_flow],   max(maximum(prob.g[line_flow]), 0.0))
        return true
    end

    MOI.set(opfmodel, Ipopt.CallbackFunction(), my_callback)
    return opfmodel
end

function store_solution!(buffer::ExaPF.PolarNetworkState, model::JuMP.Model)
    ExaPF.setvalues!(buffer, PS.ActivePower(), JuMP.value.(model[:Pg]))
    ExaPF.setvalues!(buffer, PS.ReactivePower(), JuMP.value.(model[:Qg]))
    ExaPF.setvalues!(buffer, PS.VoltageMagnitude(), JuMP.value.(model[:Vm]))
    ExaPF.setvalues!(buffer, PS.VoltageAngle(), JuMP.value.(model[:Va]))
    return
end

function main(datafile::String)
    polar = ExaPF.PolarForm(datafile)
    buffer = ExaPF.get(polar, ExaPF.PhysicalState())
    ExaPF.init_buffer!(polar, buffer)
    m = build_opf_model(polar, buffer, Ipopt.Optimizer; line_constraints=true)
    # attach_callback!(m)
    JuMP.optimize!(m)
    return m
end

