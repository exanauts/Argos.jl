# MathOptInterface wrapper
```@setup moi_wrapper
using LazyArtifacts

artifact_toml = joinpath(@__DIR__, "..", "..", "..", "Artifacts.toml")
exadata_hash = artifact_hash("ExaData", artifact_toml)
INSTANCES_DIR = joinpath(artifact_path(exadata_hash), "ExaData")

```
`Argos.jl` provides a utility to pass the nonlinear problem
encoded by a [`Argos.AbstractNLPEvaluator`](@ref) to a [MathOptInterface](https://github.com/jump-dev/MathOptInterface.jl) (MOI)
optimization problem. Hence, we can solve the
optimal power flow problem using any nonlinear optimization solver compatible
with MOI (Ipopt, Knitro, ALGENCAN, ...).

Once the MOI optimizer set up properly, passing the
optimizer to Argos simply amounts to call the function `Argos.optimize!`.
For example, solving an optimal power flow problem in the full-space
amounts to:

```@example moi_wrapper
using MathOptInterface
using Argos, Ipopt
const MOI = MathOptInterface

# Import data
datafile = joinpath(INSTANCES_DIR, "case57.m")
flp = Argos.FullSpaceEvaluator(datafile)
# Set-up MOI
optimizer = Ipopt.Optimizer()
MOI.set(optimizer, MOI.RawOptimizerAttribute("print_level"), 5)
MOI.set(optimizer, MOI.RawOptimizerAttribute("tol"), 1e-5)
# Solve
solution = Argos.optimize!(optimizer, flp)
MOI.empty!(optimizer)

```
