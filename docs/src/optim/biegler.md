```@meta
CurrentModule = Argos
```
```@setup bieglermadnlp
using LazyArtifacts
using Argos, ExaPF
using SparseArrays

artifact_toml = joinpath(@__DIR__, "..", "..", "..", "Artifacts.toml")
exadata_hash = artifact_hash("ExaData", artifact_toml)
INSTANCES_DIR = joinpath(artifact_path(exadata_hash), "ExaData")

```

# Biegler's method

Solving the OPF in the reduced-space allows to compress
drastically the size of the OPF problem, but this comes
with several downsides: (1) the power flow equations have to be solved
at each iteration, (2) we have to evaluate explicitly the
dense reduced Hessian and dense reduced Jacobian and (3) the
solution is not so accurate (tolerance set to `tol=1e-6`) as the
power flow equations are solved only approximately.

Alternatively, one can use [Biegler's method](https://epubs.siam.org/doi/abs/10.1137/0805017), which reduces the KKT system directly in the [full-space](fullspace.md)
instead of working in the reduced-space.
In exact arithmetic, this method is exactly equivalent to the
full-space method, avoiding all the [reduced-space](reducedspace.md)
machinery.

As this method is working in the full-space,
we start by instantiating a new [`FullSpaceEvaluator`](@ref):
```@example bieglermadnlp
datafile = joinpath(INSTANCES_DIR, "case9.m")
flp = Argos.FullSpaceEvaluator(datafile)

```
and wrap the resulting evaluator `flp` in a `OPFModel`:
```@example bieglermadnlp
model = Argos.OPFModel(flp)

```

## Instantiating MadNLP

Instantiating MadNLP manually to use a `BieglerKKTSystem`
is a little bit more involved, as one has to parameterize
the type directly, by defining:
```@example bieglermadnlp
KKT = Argos.BieglerKKTSystem{Float64, Vector{Int}, Vector{Float64}, Matrix{Float64}}
```
and we instantiate MadNLP with:
```@example bieglermadnlp
using MadNLP
# This syntax is a bit too involved and should be improved in the future.
madnlp_options = Dict{Symbol, Any}()
madnlp_options[:linear_solver] = MadNLP.LapackCPUSolver
opt_ipm, opt_linear, logger = MadNLP.load_options(; madnlp_options...)

KKT = Argos.BieglerKKTSystem{Float64, Vector{Int}, Vector{Float64}, Matrix{Float64}}
solver = MadNLP.MadNLPSolver{Float64, KKT}(model, opt_ipm, opt_linear; logger=logger)
```
Note that we are again using Lapack as linear solver: indeed the resulting Biegler's KKT
system is dense (we use the same condensification procedure as in the
[reduced-space](reducedspace.md)):
```@example bieglermadnlp
MadNLP.get_kkt(solver.kkt)
```

Once `solver` is instantiated, we can solve the OPF in the full-space
using the same syntax as usual:
```@repl bieglermadnlp
MadNLP.solve!(solver)
```

!!! info
    Note that we get the exact same convergence as in the [full-space](fullspace.md).

