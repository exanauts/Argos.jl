
using ExaPF
using JuMP
using Ipopt
using LinearAlgebra
using LazyArtifacts
using DelimitedFiles

using HSL_jll

const PS = ExaPF.PowerSystem

const DATA = joinpath(artifact"ExaData", "ExaData")
const SCENARIOS = joinpath(artifact"ExaData", "ExaData", "mp_demand")

constraint_index(cons::Vector{NonlinearConstraintRef{ScalarShape}}) = getfield.(JuMP.index.(cons), :value)

function PowerNetworkContingency(network::PS.PowerNetwork, remove_line)
    data = Dict{String, Array}()
    data["bus"] = copy(network.buses)
    data["branch"] = copy(network.branches)
    data["gen"] = copy(network.generators)
    data["baseMVA"] = Float64[network.baseMVA]
    data["cost"] = copy(network.costs)
    return PS.PowerNetwork(data; remove_lines=Int[remove_line])
end

function build_scopf_model(polar, buffer, lines_id, solver)
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

    Pd = buffer.pload
    Qd = buffer.qload

    cost_coefs = PS.get_costs_coefficients(pf)

    bus2gen = PS.get_bus_generators(pf.buses, pf.generators, pf.bus_to_indexes)

    # Power flow data
    nscen = length(lines_id)

    #=
        Build model
    =#

    opfmodel = Model(solver)

    # VARIABLES
    @variable(opfmodel, pg_min[i] <= Pg[i=1:ngen, k=1:nscen+1] <= pg_max[i], start=pg0[i])
    @variable(opfmodel, qg_min[i] <= Qg[i=1:ngen, k=1:nscen+1] <= qg_max[i])
    @variable(opfmodel, vm_min[i] <= Vm[i=1:nbus, k=1:nscen+1] <= vm_max[i], start=vm0[i])
    @variable(opfmodel, Va[i=1:nbus, k=1:nscen+1], start=va0[i])

    # Preventive constraints
    # u0 = uk
    gref = findfirst(isequal(pf.ref[1]), pf.gen2bus)
    for k in 1:nscen
        for g in 1:ngen
            (g == gref) && continue  # power generation at ref is a slack
            @constraint(opfmodel, Pg[g, 1] == Pg[g, k+1])
        end
        for b in [pf.pv; pf.ref]
            @constraint(opfmodel, Vm[b, 1] == Vm[b, k+1])
        end
    end

    # Reference angle
    for k in 1:nscen+1
        for b in pf.ref
            @constraint(opfmodel, Va[b, k] == 0)
        end
    end

    # Power-flow constraints
    for k in 1:nscen+1
        # TODO: this is inefficient and should be removed in a
        # future version to avoid creating a new admittance matrix
        # for each contingency.
        network = if k >= 2
            PowerNetworkContingency(pf, lines_id[k-1])
        else
            pf
        end
        Ybus = network.Ybus
        rows = Ybus.rowval
        yvals = Ybus.nzval
        g_ij = real.(yvals)
        b_ij = imag.(yvals)

        ## active
        opfmodel.ext[:active_pf] = @NLconstraint(
            opfmodel, [b=1:nbus],
            Vm[b, k] * sum(
                Vm[rows[c], k] * (g_ij[c] * cos(Va[b, k] - Va[rows[c], k]) + b_ij[c] * sin(Va[b, k] - Va[rows[c], k]))
                for c in (Ybus.colptr[b]):(Ybus.colptr[b+1]-1)
            ) == (sum(Pg[g, k] for g in get(bus2gen, b, Int[])) - Pd[b])
        )
        ## reactive
        opfmodel.ext[:reactive_pf] = @NLconstraint(
            opfmodel, [b=1:nbus],
            Vm[b, k] * sum(
                Vm[rows[c], k] * (g_ij[c] * sin(Va[b, k] - Va[rows[c], k]) - b_ij[c] * cos(Va[b, k] - Va[rows[c], k]))
                for c in (Ybus.colptr[b]):(Ybus.colptr[b+1]-1)) == (sum(Qg[g, k] for g in get(bus2gen, b, Int[])) - Qd[b])
        )
    end

    # Line constraints
    f = pf.lines.from_buses
    t = pf.lines.to_buses
    # Data
    yff_re = real.(pf.lines.Yff)
    yff_im = imag.(pf.lines.Yff)
    yft_re = real.(pf.lines.Yft)
    yft_im = imag.(pf.lines.Yft)
    ytf_re = real.(pf.lines.Ytf)
    ytf_im = imag.(pf.lines.Ytf)
    ytt_re = real.(pf.lines.Ytt)
    ytt_im = imag.(pf.lines.Ytt)
    ## from lines
    yff_abs = yff_re.^2 .+ yff_im.^2
    yft_abs = yft_re.^2 .+ yft_im.^2
    yre_fr =   yff_re .* yft_re .+ yff_im .* yft_im
    yim_fr = - yff_re .* yft_im .+ yff_im .* yft_re

    ## to lines
    ytf_abs = ytf_re.^2 .+ ytf_im.^2
    ytt_abs = ytt_re.^2 .+ ytt_im.^2
    yre_to =   ytf_re .* ytt_re .+ ytf_im .* ytt_im
    yim_to = - ytf_re .* ytt_im .+ ytf_im .* ytt_re

    for k in 1:nscen+1
        for ℓ in 1:nlines
            # Remove the lines except if base case k == 1
            if k >= 2 && ℓ == lines_id[k-1]
                continue
            end
            @NLconstraint(
                opfmodel,
                Vm[f[ℓ], k]^2 * (
                    yff_abs[ℓ] * Vm[f[ℓ], k]^2 + yft_abs[ℓ] * Vm[t[ℓ], k]^2 +
                    2.0 * Vm[f[ℓ], k] * Vm[t[ℓ], k] * (yre_fr[ℓ] * cos(Va[f[ℓ], k]-Va[t[ℓ], k]) - yim_fr[ℓ] * sin(Va[f[ℓ], k]-Va[t[ℓ], k]))
                ) <= flow_max[ℓ]
            )
            @NLconstraint(
                opfmodel,
                Vm[t[ℓ], k]^2 * (
                    ytf_abs[ℓ] * Vm[f[ℓ], k]^2 + ytt_abs[ℓ] * Vm[t[ℓ], k]^2 +
                    2.0 * Vm[f[ℓ], k] * Vm[t[ℓ], k] * (yre_to[ℓ] * cos(Va[f[ℓ], k]-Va[t[ℓ], k]) - yim_to[ℓ] * sin(Va[f[ℓ], k]-Va[t[ℓ], k]))
                ) <= flow_max[ℓ]
            )
        end
    end

    # Objective : costs in base case
    @objective(
        opfmodel,
        Min,
        sum(
            cost_coefs[g, 4] * Pg[g, 1]^2 + cost_coefs[g, 3] * Pg[g, 1] + cost_coefs[g, 2]
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

function scopf(case::String; max_contingencies=32)
    datafile = joinpath(DATA, case)
    polar = ExaPF.PolarForm(datafile)
    stack = ExaPF.NetworkStack(polar)
    instance = split(case, ".")[1]
    lines_id = readdlm(joinpath(SCENARIOS, "$(instance).Ctgs"), ',', Int)[:]

    if length(lines_id) >= max_contingencies
        @info "Capping the number of contingencies to $(max_contingencies)"
        lines_id = lines_id[1:max_contingencies]
    end

    model = build_scopf_model(polar, stack, lines_id, Ipopt.Optimizer)
    JuMP.set_attribute(model, "tol", 1e-5)
    JuMP.set_attribute(model, "hsllib", HSL_jll.libhsl_path)
    JuMP.set_attribute(model, "linear_solver", "ma27")
    JuMP.optimize!(model)
    return model
end

