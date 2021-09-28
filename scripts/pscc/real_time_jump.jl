

using Revise, JuMP, ExaPF
using DelimitedFiles

include(joinpath(@__DIR__, "..", "common.jl"))
include(joinpath(@__DIR__, "..", "jump", "jump_model.jl"))

#= PATHS =#
OPF_DATA_DIR = "/home/frapac/dev/anl/proxALM/data/"
CASE = "case118"
TS = "onehour_60"

function _load_jump_problem(case)
    datafile = joinpath(OPF_DATA_DIR, "$case.m")
    polar = ExaPF.PolarForm(datafile, CPU())
    # Buffer
    buffer = ExaPF.get(polar, ExaPF.PhysicalState())
    ExaPF.init_buffer!(polar, buffer)
    # Build model
    return build_opf_model(polar, buffer, Ipopt.Optimizer; line_constraints=false)
end

function _load_ts_data(case)
    ploads = readdlm(joinpath(OPF_DATA_DIR, "mp_demand", "$(case)_$TS.Pd"))
    qloads = readdlm(joinpath(OPF_DATA_DIR, "mp_demand", "$(case)_$TS.Qd"))
    return (ploads, qloads)
end

struct JuMPRealTimeOPF
    T::Int
    opfmodel::JuMP.Model
    ploads::Array{Float64, 2}
    qloads::Array{Float64, 2}
end

function JuMPRealTimeOPF(case::String)
    model = _load_jump_problem(case)
    ploads, qloads = _load_ts_data(case)
    baseMVA = model.ext[:exapf].baseMVA
    horizon = size(ploads, 2)
    return JuMPRealTimeOPF(horizon, model, ploads ./ baseMVA, qloads ./ baseMVA)
end

function solve!(opf::JuMPRealTimeOPF, t::Int)
    @assert 1 <= t <= opf.T
    model = opf.opfmodel
    pd = opf.ploads[:, t]
    qd = opf.qloads[:, t]
    JuMP.fix.(model[:Pd], pd)
    JuMP.fix.(model[:Qd], qd)
    JuMP.optimize!(model)
end

function solve!(opf::JuMPRealTimeOPF)
    model = opf.opfmodel
    nbus = length(model[:Vm])
    obj_vals = zeros(opf.T)
    vmag_vals = zeros(nbus, opf.T)
    for t in 1:opf.T
        solve!(opf, t)
        vmag_vals[:, t] .= JuMP.value.(model[:Vm])
        obj_vals[t] = JuMP.objective_value(model)
    end
    return (obj_vals, vmag_vals)
end

struct ExaRealTimeOPF
    T::Int
    aug::ExaOpt.AugLagEvaluator
    ploads::Array{Float64, 2}
    qloads::Array{Float64, 2}
    line_constraints::Bool
end

function ExaRealTimeOPF(case::String; line_constraints=false)
    datafile = joinpath(OPF_DATA_DIR, "$case.m")
    aug = build_problem(datafile)
    ploads, qloads = _load_ts_data(case)

    polar = ExaOpt.backend(aug)
    baseMVA = polar.network.baseMVA
    horizon = size(ploads, 2)
    return ExaRealTimeOPF(horizon, aug, ploads ./ baseMVA, qloads ./ baseMVA, line_constraints)
end

dual_variable(model, key) = -JuMP.dual.(JuMP.LowerBoundRef.(model[key])) .- JuMP.dual.(JuMP.UpperBoundRef.(model[key]))

function warmstart!(opf::ExaRealTimeOPF)
    polar = ExaOpt.backend(opf.aug)
    buffer = get(opf.aug, ExaPF.PhysicalState())
    m = build_opf_model(polar, buffer, Ipopt.Optimizer; line_constraints=opf.line_constraints)
    JuMP.optimize!(m)
    # Primal solution
    store_solution!(buffer, m)
    # Dual solution (vmag, pg, qg)
    # pv = polar.indexing.index_pv
    # pq = polar.indexing.index_pq
    # sl = polar.indexing.index_ref
    # λv = dual_variable(m, :Vm)[pq]
    # λp = dual_variable(m, :Pg)[polar.indexing.index_ref_to_gen]
    # λq = dual_variable(m, :Qg)
    # opf.aug.λ .= [λv; λp; λq]

    return
end

function solve!(opf::ExaRealTimeOPF; max_iter=100)
    optimizer = MadNLP.Optimizer(linear_solver=MadNLPLapackCPU)
    MOI.set(optimizer, MOI.RawParameter("tol"), 1e-5)
    MOI.set(optimizer, MOI.RawParameter("print_level"), MadNLP.DEBUG)
    MOI.set(optimizer, MOI.RawParameter("max_iter"), max_iter)
    MOI.set(optimizer, MOI.RawParameter("mu_init"), 1e-7)
    return @time ExaOpt.optimize!(optimizer, opf.aug)
end

function bsolve!(opf::ExaRealTimeOPF)
    ExaOpt.reset!(opf.aug)
    warmstart!(opf)
    solve!(opf)
end

function solve!(opf::ExaRealTimeOPF, t::Int)
    pd = opf.ploads[:, t]
    qd = opf.qloads[:, t]
    ExaOpt.setvalues!(opf.aug, PS.ActiveLoad(), pd)
    ExaOpt.setvalues!(opf.aug, PS.ReactiveLoad(), qd)
    warmstart!(opf)
end

function set_loads!(opf::ExaRealTimeOPF, t::Int)
    pd = opf.ploads[:, t]
    qd = opf.qloads[:, t]
    ExaOpt.setvalues!(opf.aug, PS.ActiveLoad(), pd)
    ExaOpt.setvalues!(opf.aug, PS.ReactiveLoad(), qd)
    return
end

function _solve_qp!(
    qp::ExaOpt.QuadraticModel;
    max_iter=100, linear_solver=MadNLPLapackCPU,
)
    mnlp = MadNLP.NonlinearProgram(qp)
    options = Dict{Symbol, Any}(
        :tol=>1e-4, :max_iter=>max_iter,
        :kkt_system=>MadNLP.DENSE_KKT_SYSTEM,
        :print_level=>MadNLP.ERROR,
        :linear_solver=>linear_solver,
        :hessian_constant=>true,
        :jacobian_constant=>true,
    )
    ipp = MadNLP.Solver(mnlp; option_dict=options)
    MadNLP.optimize!(ipp)
    return ipp.x
end

function tracking_algorithm!(opf::ExaRealTimeOPF)
    aug = opf.aug
    # Get initial point
    xₖ = ExaOpt.initial(aug)

    qp = ExaOpt.QuadraticModel(aug)

    T = opf.T
    obj_vals = zeros(T)
    for t in 1:T
        # Update loads to time t
        set_loads!(opf, t)

        # Update quadratic model with new setpoint
        ExaOpt.refresh!(qp, xₖ)

        # Update primal
        xₖ .= _solve_qp!(qp)
        # Update dual
        ExaOpt.update!(aug, xₖ)
        ExaOpt.update_multipliers!(aug)

        # Update solution
        obj_vals[t] = ExaOpt.objective(aug, xₖ)
    end

    return obj_vals
end

kpi(a, b) = norm(abs.(a .- b) / max.(1.0, b), Inf)

#=
    Unit-tests
=#
function test_qp(
    aug::ExaOpt.AugLagEvaluator; max_iter=100, scaling=true,
    linear_solver=MadNLPLapackCPU,
    inertia=MadNLP.INERTIA_AUTO,
)
    qp = ExaOpt.QuadraticModel(aug)
    mnlp = MadNLP.NonlinearProgram(qp)

    options = Dict{Symbol, Any}(
        :tol=>1e-4, :max_iter=>max_iter,
        :nlp_scaling=>scaling,
        :inertia_correction_method=>inertia,
        :kkt_system=>MadNLP.DENSE_KKT_SYSTEM,
        :print_level=>MadNLP.DEBUG,
        :linear_solver=>linear_solver,
        :lapackcpu_algorithm=>MadNLPLapackCPU.CHOLESKY,
        :hessian_constant=>true,
        :jacobian_constant=>true,
        :mu_init=>1e-4,
    )
    ipp = MadNLP.Solver(mnlp; option_dict=options)
    MadNLP.optimize!(ipp)
    return (qp, ipp)
end

function rto(datafile)
    opf_jump = JuMPRealTimeOPF(datafile)
    # Compute reference with Ipopt
    obj_ref, v_ref = solve!(opf_jump)

    opf_exa = ExaRealTimeOPF(datafile)
    opf_exa.aug.ρ = 0.1 # small penalty is better
    warmstart!(opf_exa)
    obj_res = tracking_algorithm!(opf_exa)

    return obj_ref, obj_res
end

