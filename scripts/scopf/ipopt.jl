
include("common.jl")

using Ipopt
using MathOptInterface
const MOI = MathOptInterface

# using HSL_jll

datafile = "/home/fpacaud/dev/matpower/data/case118.m"
lines = readdlm(joinpath(SCENARIOS, "case118_onehour_60.Ctgs"), ',', Int)[:]

model = ExaPF.PolarForm(datafile)
nbus = PS.get(model, PS.NumberOfBuses())
nlines = PS.get(model, PS.NumberOfLines())

lines = filter_islanding(model.network, lines)[1:8]
blk = build_scopf_model(model, lines; use_gpu=true)

optimizer = Ipopt.Optimizer()
MOI.set(optimizer, MOI.RawOptimizerAttribute("print_level"), 5)
MOI.set(optimizer, MOI.RawOptimizerAttribute("tol"), 1e-5)
# MOI.set(optimizer, MOI.RawOptimizerAttribute("hsllib"), HSL_jll.libhsl_path)
# MOI.set(optimizer, MOI.RawOptimizerAttribute("linear_solver"), "ma57")

solution = Argos.optimize!(optimizer, blk.nlp)
MOI.empty!(optimizer)
