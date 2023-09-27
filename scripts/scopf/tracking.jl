

using Printf
using MadNLP
using ExaPF
using Argos

include("common.jl")
include("screening.jl")

function tracking(model::ExaPF.PolarForm, lines, kkt, horizon; use_gpu=false)
    blk = build_scopf_model(model, lines; use_gpu=use_gpu)
    ev = Argos.backend(blk)

    # Warm-up
    solver = build_madnlp(blk, kkt; print_level=MadNLP.DEBUG)
    MadNLP.solve!(solver; max_iter=1)

    # Initial solve
    solver = build_madnlp(blk, kkt; print_level=MadNLP.ERROR)
    MadNLP.solve!(solver)

    cumulative_times = Float64[]
    cumulative_iterations = Int[]
    # Track solution (every 5mn)
    for t in 1:horizon
        # Update loads
        ev.stack.pload .*= .99
        ev.stack.qload .*= .99
        x = Argos.initial(ev)
        Argos.update!(ev, x)
        copyto!(blk.meta.x0, x)
        MadNLP.solve!(solver; mu_init=1e-6)
        pg = sum(ev.stack.pgen)
        push!(cumulative_times, solver.cnt.total_time)
        push!(cumulative_iterations, solver.cnt.k)
        @printf(" %4d %4d %12.5e %12.5e %10.4f %3d\n", t, solver.cnt.k, solver.obj_val, pg, solver.cnt.total_time, Int(solver.status))
    end
    println("TOTAL TIME (s): ", solver.cnt.total_time)
    return solver, cumulative_times, cumulative_iterations
end

model = PolarForm("/home/fpacaud/dev/matpower/data/case1354pegase.m")
kkt = Argos.FullSpace()
lines = readdlm(joinpath(SCENARIOS, "case1354pegase_onehour_60.Ctgs"), ',', Int)[:]
lines = filter_islanding(model.network, lines)[1:16]
solver, cum_times, cum_it = tracking(model, lines, kkt, 13; use_gpu=true)
