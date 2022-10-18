```@meta
CurrentModule = Argos
```
```@setup reducedmadnlp
using LazyArtifacts
using Argos, ExaPF
using SparseArrays

artifact_toml = joinpath(@__DIR__, "..", "..", "..", "Artifacts.toml")
exadata_hash = artifact_hash("ExaData", artifact_toml)
INSTANCES_DIR = joinpath(artifact_path(exadata_hash), "ExaData")

```

# Reduced-space OPF

Solving the OPF in the reduced-space (aka *reduce-then-linearize*) follows the same
procedure as in the [full-space](fullspace.md).

We start by instantiating a [`ReducedSpaceEvaluator`](@ref):
```@example reducedmadnlp
datafile = joinpath(INSTANCES_DIR, "case9.m")
red = Argos.ReducedSpaceEvaluator(datafile)

```

## Instantiating MadNLP
As [before](fullspace.md), we wrap the `ReducedSpaceEvaluator`
object `red` with an `OPFModel`:
```@example reducedmadnlp
model = Argos.OPFModel(red)

```

### Why we should not use MadNLP's default parameters?
Suppose now we instantiate a default MadNLP instance:
```@example reducedmadnlp
using MadNLP
solver = MadNLP.MadNLPSolver(model)

```
By default, MadNLP is using a sparse data structure. This is
not appropriate to handle a `ReducedSpaceEvaluator`, which generates
a dense Jacobian and a dense reduced Hessian. Indeed, by default
MadNLP generates a sparse KKT system with a significant
number of nonzeroes:
```@example reducedmadnlp
MadNLP.get_kkt(solver.kkt)

```

### How to parameterize MadNLP to use dense data structure?
Instead, we can parameterize MadNLP to use a dense KKT system.
The first option is to use a `DENSE_KKT_SYSTEM` in conjunction with
a dense linear solver (as Lapack):
```@example reducedmadnlp
solver = MadNLP.MadNLPSolver(
    model;
    kkt_system=MadNLP.DENSE_KKT_SYSTEM,
    linear_solver=LapackCPUSolver,
)
MadNLP.get_kkt(solver.kkt)

```
The associated KKT system is now represented as a dense matrix, more appropriate
for a dense problem. However, the generated KKT system is still too large, as its
size is proportional to both the number of variables and the number of constraints. This
approach is not tractable on larger problems.
Fortunately, MadNLP allows to [condense the KKT system](https://madnlp.github.io/MadNLP.jl/dev/lib/kkt/#MadNLP.AbstractCondensedKKTSystem)
using a Schur complement approach. By doing so, the size of the KKT system is
only proportional to the number of variables (here, `5`):
```@example reducedmadnlp
solver = MadNLP.MadNLPSolver(
    model;
    kkt_system=MadNLP.DENSE_CONDENSED_KKT_SYSTEM,
    linear_solver=LapackCPUSolver,
)
MadNLP.get_kkt(solver.kkt)

```
This alternative approach permits to significantly compress the size
of the KKT system, and is the approach used by default in `run_opf`.

Once the problem is written in condensed form inside MadNLP, one
can solve the OPF in the reduced-space with:
```@repl reducedmadnlp
stats = MadNLP.solve!(solver; tol=1e-6)

```

!!! info
We recommend changing the default tolerance to be above the tolerance
    of the Newton-Raphson used inside [`ReducedSpaceEvaluator`](@ref). Indeed,
    the power flow is solved only approximately, leading to slightly inaccurate
    evaluations and derivatives, impacting the convergence of the interior-point
    algorithm. In general, we recommend setting `tol=1e-5`.

!!! info
    Here, we are using Lapack on the CPU to solve the condensed
    KKT system at each iteration of the interior-point algorithm.
    However, if an NVIDIA GPU is available, we recommend using
    a CUDA-accelerated Lapack version, more efficient than
    the default Lapack. If `MadNLPGPU` is installed, this amounts to
    ```julia
    using MadNLPGPU
    solver = MadNLP.MadNLPSolver(
        model;
        kkt_system=MadNLP.DENSE_CONDENSED_KKT_SYSTEM,
        linear_solver=LapackGPUSolver,
        tol=1e=5,
    )
    MadNLP.solve!(solver)
    ```


## Querying the solution
As [before](fullspace.md), we can query the solution returned
by MadNLP with:
```@example reducedmadnlp
stats.objective
```
and
```@example reducedmadnlp
stats.solution
```
Or, alternatively, one can look directly at the values
in the `stack` stored inside `red`:
```@example reducedmadnlp
stack = red.stack
[stack.vmag stack.vang]

```

