
using Revise

include("common.jl")

case = "1354pegase"
datafile = "../matpower/data/case$(case).m"
ctgs = readdlm(joinpath(SCENARIOS, "case$(case)_onehour_60.Ctgs"), ',', Int)[:]
# ctgs = readdlm(joinpath(SCENARIOS, "case$(case).Ctgs"), ',', Int)[:]

function _build_madnlp_gpu(blk::Argos.OPFModel)
    madnlp_options = Dict{Symbol, Any}()
    madnlp_options[:linear_solver] = LapackGPUSolver
    madnlp_options[:lapack_algorithm] = MadNLP.CHOLESKY
    madnlp_options[:dual_initialized] = true
    madnlp_options[:max_iter] = 250
    madnlp_options[:print_level] = MadNLP.DEBUG
    madnlp_options[:tol] = 1e-5
    opt_ipm, opt_linear, logger = MadNLP.load_options(; madnlp_options...)
    KKT = Argos.BieglerKKTSystem{Float64, CuVector{Int}, CuVector{Float64}, CuMatrix{Float64}}
    return MadNLP.MadNLPSolver{Float64, KKT}(blk, opt_ipm, opt_linear; logger=logger)
end

function test_1(datafile, line)
    network = PS.PowerNetwork(datafile; remove_lines=[lines])
    network = PS.PowerNetwork(datafile)
    println(line)
    network.lines.Yff[line] = 0.0im
    network.lines.Yft[line] = 0.0im
    network.lines.Ytf[line] = 0.0im
    network.lines.Ytt[line] = 0.0im

    model = ExaPF.PolarForm(network, ExaPF.CPU())
    stack = ExaPF.NetworkStack(model)

    conv = ExaPF.run_pf(model, stack; verbose=1)
    return conv
end

function test_2(model, lines)
    nlp = build_model(model, lines; use_gpu=false)
    ips = MadNLP.MadNLPSolver(nlp; linear_solver=Ma27Solver, tol=1e-5, print_level=MadNLP.DEBUG)
    MadNLP.solve!(ips)
    return ips
end

function test_3(model, lines)
    model = ExaPF.PolarForm(datafile, CPU())
    nlp = build_model(model, lines; use_gpu=true)
    ips = _build_madnlp_gpu(nlp)
    MadNLP.solve!(ips)
    return ips
end

function test_4(datafile, lines)
    contingencies = [ExaPF.LineContingency(l) for l in lines]
    nblk = length(contingencies) + 1
    blk = ExaPF.BlockPolarForm(datafile, nblk)
    stack = ExaPF.NetworkStack(blk)

    basis = ExaPF.PolarBasis(blk)
    pf = ExaPF.PowerFlowBalance(blk, contingencies) ∘ basis

    jac_pf = ExaPF.ArrowheadJacobian(blk, pf, State())
    ExaPF.jacobian!(jac_pf, stack)
    # Solve power flow equations
    solver = ExaPF.NewtonRaphson(; verbose=1)
    conv = ExaPF.nlsolve!(solver, jac_pf, stack)
    return conv
end

model = ExaPF.PolarForm(datafile, CPU())
nlines = PS.get(model, PS.NumberOfLines())
@info "Screen contingencies"


# sctgs = g[1:4]
@info "Solve"
# conv = test_2(model, sctgs)
# conv = test_3(model, sctgs)
# ips = test_1(datafile, 4)
