

using Printf
using MadNLP
using Argos

include("common.jl")

function build_solver_cpu(blk::Argos.OPFModel)
    return MadNLP.MadNLPSolver(
        blk;
        dual_initialized=true,
        linear_solver=Ma27Solver,
        max_wall_time=1800.0,
        max_iter=250,
        print_level=MadNLP.ERROR,
        tol=1e-3,
    )
end

function build_solver_gpu(blk::Argos.OPFModel)
    madnlp_options = Dict{Symbol, Any}()
    madnlp_options[:linear_solver] = LapackGPUSolver
    madnlp_options[:lapack_algorithm] = MadNLP.CHOLESKY
    madnlp_options[:dual_initialized] = true
    madnlp_options[:max_iter] = 250
    madnlp_options[:print_level] = MadNLP.ERROR
    madnlp_options[:tol] = 1e-3
    opt_ipm, opt_linear, logger = MadNLP.load_options(; madnlp_options...)
    KKT = Argos.BieglerKKTSystem{Float64, CuVector{Int}, CuVector{Float64}, CuMatrix{Float64}}
    return MadNLP.MadNLPSolver{Float64, KKT}(blk, opt_ipm, opt_linear; logger=logger)
end

function build_solver(blk::Argos.OPFModel, gpu_solver::Bool)
    gpu_solver ? build_solver_gpu(blk) : build_solver_cpu(blk)
end


function tracking(model::ExaPF.PolarForm, nscen, T; gpu_ad=false, gpu_solver=false)
    blk = build_model(model, nscen; use_gpu=gpu_ad)
    stoch = Argos.backend(blk)

    # Warm-up
    solver = build_solver(blk, gpu_solver)
    MadNLP.solve!(solver; max_iter=2)

    # Initial solve
    solver = build_solver(blk, gpu_solver)
    MadNLP.solve!(solver)

    # Track solution (every 5mn)
    for t in 1:T
        # Update loads
        stoch.stack.pload .*= .99
        stoch.stack.qload .*= .99
        x = Argos.initial(stoch)
        Argos.update!(stoch, x)
        copyto!(blk.meta.x0, x)
        MadNLP.solve!(solver; mu_init=1e-6)
        pg = sum(stoch.stack.pgen)
        @printf(" %4d %4d %12.5e %12.5e %3d\n", t, solver.cnt.k, solver.obj_val, pg, Int(solver.status))
    end
    println("TOTAL TIME (s): ", solver.cnt.total_time)
    return solver
end

ips = tracking(model, 8, 12; gpu_ad=true)
