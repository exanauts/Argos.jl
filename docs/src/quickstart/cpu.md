# Argos on CPU (default)
```@setup quickstart_cpu
using LazyArtifacts

artifact_toml = joinpath(@__DIR__, "..", "..", "..", "Artifacts.toml")
exadata_hash = artifact_hash("ExaData", artifact_toml)
INSTANCES_DIR = joinpath(artifact_path(exadata_hash), "ExaData")

```

Unless explicitly specified, Argos solves the OPF
on the CPU. Argos provides 3 different methods to solve
the OPF, all relying on the interior-point method.

---

| Method            | Derivatives | KKT system |
| ----------------- | ----------- | ---------- |
| **Full-space**      | Sparse      | Sparse     |
| **Biegler**         | Sparse      | Dense      |
| **Dommel & Tinney** | Dense       | Dense      |

---

- **Full-space** and **Biegler** are equivalent in exact arithmetic. **Full-space** solves the original (sparse) KKT system whereas **Biegler** uses a reduction method to find an equivalent (dense) linear system.
- **Dommel & Tinney** is a port of the [classical algorithm of Dommel and Tinney](https://ieeexplore.ieee.org/abstract/document/4073461?casa_token=hJTPXuWhZPwAAAAA:eafEn2hG-_oQG5M-je0hGC86h4sZfdElvf0oBPqlfeNSbWWH1ckLC2RICB5EJO4M60SvAb-jaQ) originally proposed in 1968. It exploits the structure of the power flow equations to optimize only with relation to the OPF's degrees of freedom (the *control*, associated to the voltage at the PV nodes and the power generations). The reduction is the same as in **Biegler**, but **Dommel & Tinney** comprises an additional projection step to recover the dependent variables (the *state*). Currently, **Dommel & Tinney** uses the Newton-Raphson algorithm implemented in ExaPF to compute the projection step.

All three methods are described in detail in the
[Argos paper](https://arxiv.org/abs/2203.11875).

First, we set up the current Julia environment and choose
an OPF instance in our artifact [ExaData](https://github.com/exanauts/ExaData):
```@example quickstart_cpu
using Argos, MadNLP
datafile = joinpath(INSTANCES_DIR, "case118.m")

```

## Full-space method

!!! tip
    At each iteration of the algorithm,
    `FullSpace` solves the KKT system with a sparse linear solver.
    By default, MadNLP is using Umfpack, but we recommend installing
    [MadNLPHSL](https://madnlp.github.io/MadNLP.jl/dev/installation/#HSL-linear-solver)
    and uses ma27 (`linear_solver=Ma27Solver`) or ma57 (`linear_solver=Ma57Solver`).


```@repl quickstart_cpu
Argos.run_opf(datafile, Argos.FullSpace());

```

## Biegler's method (linearize-then-reduce)

!!! tip
    - Biegler's reduction condenses and reduces the KKT linear system to
      a dense linear system whose size is given by the number of degrees
      of freedom in the problem. We recommend factorizing the resulting
      system with the Cholesky factorization shipped with Lapack.
    - Note we obtain exactly the same convergence as with the previous `FullSpace` method, as the two methods are equivalent.

```@repl quickstart_cpu
Argos.run_opf(datafile, Argos.BieglerReduction(); lapack_algorithm=MadNLP.CHOLESKY);

```

## Dommel & Tinney's method (reduce-then-linearize)

!!! tip
    - `DommelTinney` works in the reduced space, and
      the associated formulation has less variable
      than in the full space (107 versus 288).
    - The reduced Jacobian and reduced Hessian are dense,
      so `DommelTinney` can potentially eat a lot of memory on
      the largest instances.
    - As with `BieglerReduction`, we recommend using Lapack with
      the Cholesky factorization to solve the KKT system.
    - Note that we have to increase MadNLP's tolerance (parameter `tol`)
      as we cannot optimize below the tolerance of the Newton-Raphson
      employed under the hood (`1e-10` by default).

```@repl quickstart_cpu
Argos.run_opf(datafile, Argos.DommelTinney(); tol=1e-5);

```

