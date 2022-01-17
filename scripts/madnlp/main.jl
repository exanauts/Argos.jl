
using Revise
using SuiteSparse
using ExaPF
using MadNLP
using MathOptInterface
using Argos
using CUDA

const MOI = MathOptInterface

# Remove iterative refinement in UMFPACK
SuiteSparse.UMFPACK.umf_ctrl[8] = 0.0

if CUDA.has_cuda_gpu()
    using MadNLPGPU
    include(joinpath(dirname(pathof(Argos)), "..", "test", "cusolver.jl"))
end

function solve_auglag_moi(aug; linear_solver=MadNLPLapackCPU, max_iter=20, penalty=0.1, rate=10.0)
    Argos.reset!(aug)
    aug.tracker = Argos.NLPTracker(aug)
    options = Argos.AugLagOptions(;
        max_iter=max_iter,
        max_inner_iter=1000,
        α0=1.0,
        rate=rate,
        ωtol=1e-6,
        verbose=1,
        ε_dual=1e-2,
        ε_primal=1e-5,
    )
    optimizer = MadNLP.Optimizer(
        linear_solver=linear_solver, print_level=MadNLP.ERROR, max_iter=500,
    )
    solver = Argos.AuglagSolver(optimizer, options)

    x0 = Argos.initial(aug)
    aug.ρ = penalty # update penalty in Evaluator

    return Argos.optimize!(solver, aug, x0)
end

function solve_auglag_madnlp(aug; linear_solver=MadNLPLapackCPU, max_iter=10, penalty=10.0, rate=10.0)
    options = Argos.AugLagOptions(;
        max_iter=max_iter,
        max_inner_iter=100,
        α0=1.0,
        rate=rate,
        ωtol=1e-5,
        verbose=1,
        ε_dual=1e-2,
        ε_primal=1e-5,
    )
    Argos.reset!(aug)
    aug.ρ = penalty # update penalty in Evaluator
    mnlp = Argos.ExaNLPModel(aug)
    madnlp_options = Dict{Symbol, Any}(
        :tol=>1e-5,
        :kkt_system=>MadNLP.DENSE_KKT_SYSTEM,
        :linear_solver=>linear_solver,
        :print_level=>MadNLP.ERROR,
    )
    ipp = MadNLP.InteriorPointSolver(mnlp; option_dict=madnlp_options)
    solver = Argos.AuglagSolver(ipp, options)

    x0 = Argos.initial(aug)

    return Argos.optimize!(solver, aug, x0)
end

function solve_auglag_madnlp_schur(aug; linear_solver=MadNLPLapackCPU, max_iter=10, penalty=10.0, rate=10.0)
    options = Argos.AugLagOptions(;
        max_iter=max_iter,
        max_inner_iter=100,
        α0=1.0,
        rate=rate,
        ωtol=1e-5,
        verbose=1,
        ε_dual=1e-2,
        ε_primal=1e-5,
    )
    Argos.reset!(aug)
    mnlp = Argos.ExaNLPModel(aug)
    madnlp_options = Dict{Symbol, Any}(:tol=>1e-5,
                                       :kkt_system=>MadNLP.DENSE_KKT_SYSTEM,
                                       :linear_solver=>linear_solver,
                                       :print_level=>MadNLP.ERROR)
    madopt = MadNLP.Options(linear_solver=linear_solver)
    MadNLP.set_options!(madopt, madnlp_options)
    KKT = Argos.MixedAuglagKKTSystem{Float64, CuVector{Float64}, CuMatrix{Float64}}
    ipp = MadNLP.InteriorPointSolver{KKT}(mnlp, madopt; option_linear_solver=madnlp_options)
    solver = Argos.AuglagSolver(ipp, options)

    x0 = Argos.initial(aug)
    aug.ρ = penalty # update penalty in Evaluator

    return Argos.optimize!(solver, aug, x0)
end

function subproblem_moi(aug; linear_solver=MadNLPLapackCPU, max_iter=100)
    Argos.reset!(aug)
    optimizer = MadNLP.Optimizer(linear_solver=linear_solver)
    MOI.set(optimizer, MOI.RawParameter("tol"), 1e-5)
    MOI.set(optimizer, MOI.RawParameter("print_level"), MadNLP.DEBUG)
    MOI.set(optimizer, MOI.RawParameter("max_iter"), max_iter)
    solution = @time Argos.optimize!(optimizer, aug)

    return optimizer.ips
end

function subproblem_nlp(
    aug; max_iter=100, scaling=true,
    linear_solver=MadNLPLapackCPU,
    inertia=MadNLP.INERTIA_AUTO,
)
    Argos.reset!(aug)
    mnlp = Argos.ExaNLPModel(aug)
    options = Dict{Symbol, Any}(
        :tol=>1e-5, :max_iter=>max_iter,
        :nlp_scaling=>scaling,
        :inertia_correction_method=>inertia,
        :print_level=>MadNLP.DEBUG,
        :linear_solver=>linear_solver,
    )
    ipp = MadNLP.InteriorPointSolver(mnlp; option_dict=options)
    @time MadNLP.optimize!(ipp)
    return ipp
end

function subproblem_biegler(
    aug; max_iter=100, scaling=true,
    linear_solver=MadNLPLapackCPU,
    inertia=MadNLP.INERTIA_AUTO,
)
    Argos.reset!(aug)
    mnlp = Argos.ExaNLPModel(aug)
    options = Dict{Symbol, Any}(
        :tol=>1e-5, :max_iter=>max_iter,
        :nlp_scaling=>scaling,
        :inertia_correction_method=>inertia,
        :print_level=>MadNLP.DEBUG,
        :linear_solver=>linear_solver,
    )
    madopt = MadNLP.Options(linear_solver=linear_solver)
    MadNLP.set_options!(madopt, options,Dict())

    KKT = Argos.BieglerKKTSystem{Float64, Vector{Int}, Vector{Float64}, Matrix{Float64}}
    ipp = MadNLP.InteriorPointSolver{KKT}(mnlp, madopt; option_linear_solver=options)
    @time MadNLP.optimize!(ipp)
    return ipp
end

function subproblem_dense_kkt(
    aug; max_iter=100, scaling=true,
    linear_solver=MadNLPLapackCPU,
    inertia=MadNLP.INERTIA_AUTO,
)
    Argos.reset!(aug)
    mnlp = Argos.ExaNLPModel(aug)
    options = Dict{Symbol, Any}(
        :tol=>1e-5, :max_iter=>max_iter,
        :nlp_scaling=>scaling,
        :inertia_correction_method=>inertia,
        :kkt_system=>MadNLP.DENSE_KKT_SYSTEM,
        :print_level=>MadNLP.DEBUG,
        :linear_solver=>linear_solver
    )
    ipp = MadNLP.InteriorPointSolver(mnlp; option_dict=options)
    @time MadNLP.optimize!(ipp)
    return ipp
end

function subproblem_schur_kkt(
    aug; max_iter=100, scaling=true,
    linear_solver=MadNLPLapackCPU,
    inertia=MadNLP.INERTIA_AUTO,
)
    Argos.reset!(aug)
    mnlp = Argos.ExaNLPModel(aug)

    options = Dict{Symbol, Any}(
        :tol=>1e-5, :max_iter=>max_iter,
        :nlp_scaling=>scaling,
        :inertia_correction_method=>inertia,
        :print_level=>MadNLP.DEBUG,
        :kkt_system=>MadNLP.DENSE_KKT_SYSTEM,
        :linear_solver=>linear_solver
    )
    madopt = MadNLP.Options(linear_solver=linear_solver)
    MadNLP.set_options!(madopt,options,Dict())
    # Custom KKT system
    KKT = Argos.MixedAuglagKKTSystem{Float64, Vector{Float64}, Matrix{Float64}}
    ipp = MadNLP.InteriorPointSolver{KKT}(mnlp, madopt; option_linear_solver=options)
    @time MadNLP.optimize!(ipp)
    return ipp
end

function subproblem_dense_kkt_gpu(aug; max_iter=100)
    @assert CUDA.has_cuda_gpu()
    linear_solver = MadNLPLapackGPU
    Argos.reset!(aug)
    mnlp = Argos.ExaNLPModel(aug)
    options = Dict{Symbol, Any}(:tol=>1e-5, :max_iter=>max_iter,
                                :print_level=>MadNLP.DEBUG,
                                :kkt_system=>MadNLP.DENSE_KKT_SYSTEM,
                                :linear_solver=>linear_solver)
    madopt = MadNLP.Options(linear_solver=linear_solver)
    MadNLP.set_options!(madopt,options,Dict())
    # Custom KKT type
    KKT = MadNLP.DenseKKTSystem{Float64, CuVector{Float64}, CuMatrix{Float64}}
    ipp = MadNLP.InteriorPointSolver{KKT}(mnlp, madopt; option_linear_solver=options)
    @time MadNLP.optimize!(ipp)
    return ipp
end

function subproblem_schur_kkt_gpu(aug; max_iter=100)
    @assert CUDA.has_cuda_gpu()
    linear_solver = MadNLPLapackGPU
    Argos.reset!(aug)
    mnlp = Argos.ExaNLPModel(aug)
    options = Dict{Symbol, Any}(:tol=>1e-5, :max_iter=>max_iter,
                                :print_level=>MadNLP.DEBUG,
                                :kkt_system=>MadNLP.DENSE_KKT_SYSTEM,
                                :linear_solver=>linear_solver)
    madopt = MadNLP.Options(linear_solver=linear_solver)
    MadNLP.set_options!(madopt,options,Dict())
    # Custom KKT type
    KKT = Argos.MixedAuglagKKTSystem{Float64, CuVector{Float64}, CuMatrix{Float64}}
    ipp = MadNLP.InteriorPointSolver{KKT}(mnlp, madopt; option_linear_solver=options)
    @time MadNLP.optimize!(ipp)
    return ipp
end

function subproblem_condensed_kkt(
    aug; max_iter=100, scaling=true,
    linear_solver=MadNLPLapackCPU,
    inertia=MadNLP.INERTIA_AUTO,
)
    Argos.reset!(aug)
    mnlp = Argos.ExaNLPModel(aug)
    options = Dict{Symbol, Any}(
        :tol=>1e-5, :max_iter=>max_iter,
        :nlp_scaling=>scaling,
        :inertia_correction_method=>inertia,
        :kkt_system=>MadNLP.CONDENSED_KKT_SYSTEM,
        :print_level=>MadNLP.DEBUG,
        :linear_solver=>linear_solver,
        # :lapackcpu_algorithm=>MadNLPLapackCPU.CHOLESKY,
    )
    ipp = MadNLP.InteriorPointSolver(mnlp; option_dict=options)
    @time MadNLP.optimize!(ipp)
    return ipp
end

function subproblem_condensed_kkt_gpu(aug; max_iter=100, scaling=true)
    @assert CUDA.has_cuda_gpu()
    linear_solver = MadNLPLapackGPU
    Argos.reset!(aug)
    mnlp = Argos.ExaNLPModel(aug)
    options = Dict{Symbol, Any}(
        :tol=>1e-5, :max_iter=>max_iter,
        :print_level=>MadNLP.DEBUG,
        :nlp_scaling=>scaling,
        :kkt_system=>MadNLP.DENSE_KKT_SYSTEM,
        :linear_solver=>linear_solver,
        :lapackgpu_algorithm=>MadNLPLapackGPU.CHOLESKY,
    )
    madopt = MadNLP.Options(linear_solver=linear_solver)
    MadNLP.set_options!(madopt, options, Dict())
    # Custom KKT type
    KKT = MadNLP.DenseCondensedKKTSystem{Float64, CuVector{Float64}, CuMatrix{Float64}}
    ipp = MadNLP.InteriorPointSolver{KKT}(mnlp, madopt; option_linear_solver=options)
    @time MadNLP.optimize!(ipp)
    return ipp
end
