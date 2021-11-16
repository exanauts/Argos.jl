# ExaPF

[`Argos.jl`](https://github.com/exanauts/ExaPF-Opt.jl) is a
package extending [`ExaPF.jl`](https://github.com/exanauts/ExaPF-Opt.jl)
for the resolution of large-scale optimal power flow (OPF) problems
on exascale architectures.

[`Argos.jl`](https://github.com/exanauts/ExaPF-Opt.jl) implements
a generic interface for the various callbacks needed by nonlinear
optimization solvers. The callbacks are defined inside `Evaluators`
structure, each `Evaluator` having its particular use-case.
Among these evaluators, the `[`ReducedSpaceEvaluator`](@ref)` uses
the fast power flow solver implemented in [`ExaPF.jl`](https://github.com/exanauts/ExaPF-Opt.jl)
to run the optimization directly in the reduced space induced by the
power flow equations.


## Table of contents

### Manual

```@contents
Pages = [
    "man/evaluators.md",
]
Depth = 1
```

### Library

```@contents
Pages = [
    "lib/evaluators.md",
]
Depth = 1
```

## Funding

This research was supported by the Exascale Computing Project (17-SC-20-SC), a joint project of the U.S. Department of Energy’s Office of Science and National Nuclear Security Administration, responsible for delivering a capable exascale ecosystem, including software, applications, and hardware technology, to support the nation’s exascale computing imperative.
