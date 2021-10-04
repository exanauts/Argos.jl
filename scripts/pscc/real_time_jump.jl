

using Revise, JuMP, ExaPF
using DelimitedFiles

include(joinpath(@__DIR__, "..", "common.jl"))
include(joinpath(@__DIR__, "..", "jump", "jump_model.jl"))

#= PATHS =#
OPF_DATA_DIR = "/home/frapac/dev/anl/proxALM/data/"
OPF_DATA_DIR = "/home/fpacaud/exa/proxALM/data/"
CASE = "case118"
TS = "onehour_60"

kpi(a, b) = norm(abs.(a .- b) / max.(1.0, b), Inf)

function _load_jump_problem(case)
    datafile = joinpath(OPF_DATA_DIR, "$case.m")
    polar = ExaPF.PolarForm(datafile, CPU())
    # Buffer
    buffer = ExaPF.get(polar, ExaPF.PhysicalState())
    ExaPF.init_buffer!(polar, buffer)
    # Build model
    return build_opf_model(polar, buffer, Ipopt.Optimizer; line_constraints=false)
end

function _load_ts_data(case; perturb=true)
    ploads = readdlm(joinpath(OPF_DATA_DIR, "mp_demand", "$(case)_$TS.Pd"))
    qloads = readdlm(joinpath(OPF_DATA_DIR, "mp_demand", "$(case)_$TS.Qd"))

    if perturb
        ploads[:, 3:end] .*= 0.8
        qloads[:, 3:end] .*= 0.8
    end
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
    npg = length(model[:Pg])
    obj_vals = zeros(opf.T)
    pg_vals = zeros(npg, opf.T)
    for t in 1:opf.T
        solve!(opf, t)
        pg_vals[:, t] .= JuMP.value.(model[:Pg])
        obj_vals[t] = JuMP.objective_value(model)
    end
    return (obj_vals, pg_vals)
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
    return ExaRealTimeOPF(aug, case; line_constraints=line_constraints)
end

function ExaRealTimeOPF(aug, case::String; line_constraints=false)
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
        :tol=>1e-6, :max_iter=>max_iter,
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

function _solve_qp!(
    qp::ExaOpt.AuglagQuadraticModel;
    max_iter=100, linear_solver=MadNLPLapackGPU,
)
    mnlp = MadNLP.NonlinearProgram(qp)
    kkt = ExaOpt.MixedAuglagKKTSystem{Float64, CuVector{Float64}, CuMatrix{Float64}}(qp, Int[])
    options = Dict{Symbol, Any}(
        :tol=>1e-5, :max_iter=>max_iter,
        :kkt_system=>MadNLP.DENSE_KKT_SYSTEM,
        :print_level=>MadNLP.ERROR,
        :linear_solver=>linear_solver,
        :lapackgpu_algorithm=>MadNLPLapackGPU.CHOLESKY,
    )
    ipp = MadNLP.Solver(mnlp, kkt=kkt; option_dict=options)
    MadNLP.optimize!(ipp)
    return ipp.x |> CuArray
end

function tracking_algorithm!(opf::ExaRealTimeOPF, xₖ=ExaOpt.initial(opf.aug); schur=true, y=nothing, maxT=opf.T)
    aug = opf.aug

    ExaOpt.reset!(aug)
    ExaOpt.update!(aug, xₖ)

    if !isnothing(y)
        copyto!(aug.λ, y)
    end

    if schur
        qp = ExaOpt.AuglagQuadraticModel(aug, xₖ)
    else
        qp = ExaOpt.QuadraticModel(aug)
    end

    ngen = get(aug, PS.NumberOfGenerators())
    buffer = get(aug, ExaPF.PhysicalState())
    T = opf.T
    obj_vals = zeros(T)
    prfeas_vals = zeros(T)
    pg_vals = zeros(ngen, T)
    for t in 1:maxT
        println(t)
        # Update loads to time t
        set_loads!(opf, t)

        # Update quadratic model with new setpoint
        ExaOpt.refresh!(qp, xₖ)

        # Update primal
        x₊ = _solve_qp!(qp)
        # Update dual
        conv = ExaOpt.update!(aug, x₊)
        # if !conv.has_converged
        #     ExaOpt.reset!(aug.inner)
        #     conv = ExaOpt.update!(aug, x₊)
        #     println("Iteration $t")
        #     break
        # end
        if !conv.has_converged
            ExaOpt.reset!(aug.inner)
            dₖ = (x₊ .- xₖ)
            k = 0
            α = 0.999
            while (k < 20) && !conv.has_converged
                x₊ = xₖ + α * dₖ
                ExaOpt.reset!(aug.inner)
                conv = ExaOpt.update!(aug, x₊)
                α *= 0.8
                k += 1
            end

            if !conv.has_converged
                ExaOpt.update!(aug, xₖ)
                x₊ .= xₖ
                println("Unable to converge")
                break
            end
        end

        pg_vals[:, t] .= buffer.pgen |> Array
        xₖ .= x₊
        ExaOpt.update_multipliers!(aug)

        # Update solution
        obj_vals[t] = ExaOpt.inner_objective(aug, xₖ)
        prfeas_vals[t] = norm(aug.cons, Inf)
        if !isnothing(aug.tracker)
            ExaOpt.store!(aug, aug.tracker, xₖ)
        end
    end

    return obj_vals, prfeas_vals, pg_vals
end


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

function rto_ref(datafile)
    opf_jump = JuMPRealTimeOPF(datafile)
    # Compute reference with Ipopt
    return solve!(opf_jump)
end

function rto_exa(datafile; penalty=0.1, y=nothing)
    aug_g =
    opf_exa = ExaRealTimeOPF(datafile)
    opf_exa.aug.ρ = penalty # small penalty is better
    if !isnothing(y)
        copyto!(opf_exa.aug.λ, y)
    end
    warmstart!(opf_exa)
    obj_res = tracking_algorithm!(opf_exa)
    return obj_res
end

function test_qp_schur(aug, u; max_iter=100, tol=1e-3, inertia=true, verbose=false)
    qpaug = ExaOpt.AuglagQuadraticModel(aug, u)
    @time ExaOpt.refresh!(qpaug, u)
    mnlp = MadNLP.NonlinearProgram(qpaug)
    kkt = ExaOpt.MixedAuglagKKTSystem{Float64, CuVector{Float64}, CuMatrix{Float64}}(qpaug, Int[])
    inertia_alg = inertia ? MadNLP.INERTIA_BASED : MadNLP.INERTIA_FREE
    options = Dict{Symbol, Any}(:tol=>tol, :max_iter=>max_iter,
                                :print_level=>verbose ? MadNLP.DEBUG : MadNLP.ERROR,
                                :kkt_system=>MadNLP.DENSE_KKT_SYSTEM,
                                :linear_solver=>MadNLPLapackGPU,
                                :inertia_correction_method=>inertia_alg,
                                :lapackgpu_algorithm=>MadNLPLapackGPU.CHOLESKY,
        # :hessian_constant=>true,
        # :jacobian_constant=>true,
    )
    ipp = MadNLP.Solver(mnlp; kkt=kkt, option_dict=options)
    @time MadNLP.optimize!(ipp)
    return ipp
end

