
using Printf
using JuMP

# Tools to solve the static OPF problem
include(joinpath(dirname(@__FILE__), "static_opf.jl"))
# JuMP extension
include(joinpath(dirname(pathof(ExaOpt)), "..", "scripts", "jump", "jump_model.jl"))

MadNLP.num_variables(kkt::ExaOpt.MixedAuglagKKTSystem) = size(kkt.hess, 1)

#=
    UTILS
=#

function _load_ts_data(case)
    pd_src = joinpath(joinpath(dirname(@__FILE__), "data", "$(case)_pd.txt"))
    qd_src = joinpath(joinpath(dirname(@__FILE__), "data", "$(case)_qd.txt"))
    ploads = readdlm(pd_src)
    qloads = readdlm(qd_src)
    return (ploads, qloads)
end

kpi(a, b) = norm(abs.(a .- b) / max.(1.0, b), Inf)

#=
    JuMP
=#
function _load_jump_problem(case)
    datafile = joinpath(DATA_DIRECTORY, case)
    polar = ExaPF.PolarForm(datafile, CPU())
    # Buffer
    buffer = ExaPF.get(polar, ExaPF.PhysicalState())
    ExaPF.init_buffer!(polar, buffer)
    # Build model
    return build_opf_model(polar, buffer, Ipopt.Optimizer; line_constraints=false)
end

struct JuMPRealTimeOPF
    T::Int
    opfmodel::JuMP.Model
    ploads::Array{Float64, 2}
    qloads::Array{Float64, 2}
end

function JuMPRealTimeOPF(case::String)
    model = _load_jump_problem("$(case).m")
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

function rto_ref(datafile)
    opf_jump = JuMPRealTimeOPF(datafile)
    # Compute reference with Ipopt
    return solve!(opf_jump)
end

#=
    REAL-TIME OPF
=#
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

# Warmstart with Ipopt's solution
function warmstart!(opf::ExaRealTimeOPF)
    polar = ExaOpt.backend(opf.aug)
    buffer = get(opf.aug, ExaPF.PhysicalState())
    m = build_opf_model(polar, buffer, Ipopt.Optimizer; line_constraints=opf.line_constraints)
    JuMP.optimize!(m)
    # Primal solution
    store_solution!(buffer, m)
    return
end

function set_loads!(opf::ExaRealTimeOPF, t::Int)
    pd = opf.ploads[:, t]
    qd = opf.qloads[:, t]
    # Update loads inside Augmented Lagrangian evaluator
    ExaOpt.setvalues!(opf.aug, PS.ActiveLoad(), pd)
    ExaOpt.setvalues!(opf.aug, PS.ReactiveLoad(), qd)
    return
end

function _solve_qp!(
    qp::ExaOpt.QuadraticModel;
    max_iter=100, linear_solver=MadNLPLapackCPU,
)
    mnlp = ExaOpt.ExaNLPModel(qp)
    options = Dict{Symbol, Any}(
        :tol=>1e-6, :max_iter=>max_iter,
        :kkt_system=>MadNLP.DENSE_KKT_SYSTEM,
        :print_level=>MadNLP.ERROR,
        :linear_solver=>linear_solver,
        :hessian_constant=>true,
        :jacobian_constant=>true,
    )
    ipp = MadNLP.InteriorPointSolver(mnlp; option_dict=options)
    MadNLP.optimize!(ipp)
    return ipp.x
end

function _solve_qp!(
    qp::ExaOpt.AuglagQuadraticModel;
    max_iter=100, linear_solver=MadNLPLapackGPU,
)
    @assert CUDA.has_cuda_gpu()
    mnlp = ExaOpt.ExaNLPModel(qp)
    kkt = ExaOpt.MixedAuglagKKTSystem{Float64, CuVector{Float64}, CuMatrix{Float64}}(qp, Int[])
    options = Dict{Symbol, Any}(
        :tol=>1e-5, :max_iter=>max_iter,
        :kkt_system=>MadNLP.DENSE_KKT_SYSTEM,
        :print_level=>MadNLP.ERROR,
        :linear_solver=>linear_solver,
        :lapackgpu_algorithm=>MadNLPLapackGPU.CHOLESKY,
    )
    ipp = MadNLP.InteriorPointSolver(mnlp, kkt=kkt; option_dict=options)
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
        # Backtracking line-search
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
    mnlp = ExaOpt.ExaNLPModel(qp)

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
    ipp = MadNLP.InteriorPointSolver(mnlp; option_dict=options)
    MadNLP.optimize!(ipp)
    return (qp, ipp)
end

function test_qp_schur(
    aug::ExaOpt.AugLagEvaluator{Ev, T, VT}, u;
    max_iter=100, tol=1e-3, inertia=true, verbose=false,
    linear_solver=MadNLPLapackGPU,
) where {Ev, T, VT}
    qpaug = ExaOpt.AuglagQuadraticModel(aug, u)
    t1 = @timed ExaOpt.refresh!(qpaug, u)
    mnlp = ExaOpt.ExaNLPModel(qpaug)
    inertia_alg = inertia ? MadNLP.INERTIA_BASED : MadNLP.INERTIA_FREE
    options = Dict{Symbol, Any}(
        :tol=>tol, :max_iter=>max_iter,
        :print_level=>verbose ? MadNLP.DEBUG : MadNLP.ERROR,
        :kkt_system=>MadNLP.DENSE_KKT_SYSTEM,
        :linear_solver=>linear_solver,
        :inertia_correction_method=>inertia_alg,
        :lapackgpu_algorithm=>MadNLPLapackGPU.CHOLESKY,
    )
    madopt = MadNLP.Options(linear_solver=linear_solver)
    MadNLP.set_options!(madopt, options)
    if VT <: CuArray
        KKT = ExaOpt.MixedAuglagKKTSystem{Float64, CuVector{Float64}, CuMatrix{Float64}}
    else
        KKT = ExaOpt.MixedAuglagKKTSystem{Float64, Vector{Float64}, Matrix{Float64}}
    end
    ipp = MadNLP.InteriorPointSolver{KKT}(mnlp, madopt; option_linear_solver=options)
    t2 = @timed MadNLP.optimize!(ipp)
    return ipp, t1, t2
end

"""
    Measure the time to update the tracking control.
    Reproduce the results displayed in Table III.
"""
function pscc_time_tracking_update()
    nbatches = 250
    for case in [
        "case1354pegase.m",
        "case2869pegase.m",
        "case9241pegase.m",
    ]
        @info "Benchmark tracking update: $case"
        datafile = joinpath(DATA_DIRECTORY, case)
        aug_g = ExaOpt.instantiate_auglag_model(
            datafile;
            scale=false, line_constraints=true,
            device=CUDADevice(), nbatches=nbatches,
        )
        _set_manual_scaler!(aug_g)
        res = solve_auglag_madnlp_schur(aug_g; rate=5.0, max_iter=20)
        ipp, t1, t2 = test_qp_schur(aug_g, res.minimizer; verbose=true)
        @printf("Update Hessian:    %.3f \n", t1.time)
        @printf("Solve QP with IPM: %.3f \n", t2.time)
        @printf("    #iter:          %i\n", ipp.cnt.k)
        @printf("    Linalg (s):    %.4f\n", ipp.cnt.linear_solver_time)
        @printf("    Eval (s):      %.4f\n", ipp.cnt.eval_function_time)
        @printf("TOTAL Δt:          %.3f\n", t1.time + t2.time)
    end
end

function pscc_real_time_opf(case; line_constraints=false)
    nbatches = 250
    datafile = joinpath(DATA_DIRECTORY, "$(case).m")

    # Initiate model
    aug_g = ExaOpt.instantiate_auglag_model(
        datafile;
        scale=false, line_constraints=line_constraints,
        device=CUDADevice(), nbatches=nbatches,
    )
    _set_manual_scaler!(aug_g)

    # Run static OPF algorithm
    res = solve_auglag_madnlp_schur(aug_g; rate=5.0, max_iter=20)
    y = copy(aug_g.λ)
    x0 = copy(res.minimizer)

    opf = ExaRealTimeOPF(aug_g, case)

    # Decrease ρ
    aug_g.ρ = 0.01
    return tracking_algorithm!(opf, x0; y=y)
end

