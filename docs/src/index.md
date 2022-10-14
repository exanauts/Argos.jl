# Argos

[`Argos.jl`](https://github.com/exanauts/Argos.jl) is a
Julia package extending [`ExaPF.jl`](https://github.com/exanauts/ExaPF.jl)
for the resolution of large-scale optimal power flow (OPF) problems
on exascale architectures.

Argos is built as a prototype to demonstrate the capability
of GPU/SIMD hardware to accelerate the solution of large-scale
OPF problems.

## Quickstart
Argos is a registered package, and its installation amouts to:
```julia
pkg> add Argos
```
The function `run_opf` takes as input a path to a regular
MATPOWER file to solve the associated OPF problem:
```julia
julia> Argos.run_opf("case9.m", Argos.FullSpace())

```

## Purpose

When using the polar formulation, OPF problems formulate as
large-scale nonlinear problems (NLP), challenging to solve.
When solving such large-scale problems, the two bottlenecks are:
1. The evaluation of the nonlinear first- and second-order derivatives.
2. The solution of the Karush-Kuhn-Tucker (KKT) system inside the optimization solver.

The first point (1) is alleviated by using [ExaPF](https://github.com/exanauts/ExaPF.jl), a power-system
modeler using automatic-differentation (AD) to evaluate the derivatives
efficiently in a vectorized fashion. Notably, ExaPF allows to deport
the evaluation of the derivatives on **CUDA GPU**.

The second point (2) is usually solved by using efficient sparse
linear solvers (as [Pardiso](https://www.pardiso-project.org/) or [HSL](https://www.hsl.rl.ac.uk/)). Unfortunately, it is well-known that sparse linear
solvers are difficult to port on GPU and SIMD architectures,
as illustrated in [this paper](https://www.sciencedirect.com/science/article/pii/S0167819121001125).
As such, Argos is tightly integrated with the interior-point
solver [MadNLP](https://github.com/MadNLP/MadNLP.jl) to exploit the structure
of the optimal power flow problem in a *condense and reduce* approach.

Argos is only a prototype, and is nowhere as mature
as established packages as [MATPOWER](matpower.org/)
or [PowerModels.jl](https://github.com/lanl-ansi/PowerModels.jl).
However, Argos is building on the flexibility provided by the
Julia language, and we believe it is package well-suited for research
purpose and advanced users.

## Acknowledgements

Argos development was supported by the Exascale Computing Project (ECP).
The development of Argos has been greatly facilited by the flexibility
offered by the [Julia language](julialang.org/) and its excellent
[GPU ecosystem](https://juliagpu.org/).
Notably, Argos makes an extensive use of [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl/)
and [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl).
Argos is tightly integrated with the solver
[MadNLP](https://github.com/MadNLP/MadNLP.jl), which was originally
developed at Los Alamos National Laboratory.


## Citing Argos

If you use Argos in your research, we would
greatly appreciate your citing it.
```
@article{pacaud2022condensed,
  title={Condensed interior-point methods: porting reduced-space approaches on GPU hardware},
  author={Pacaud, Fran{\c{c}}ois and Shin, Sungho and Schanen, Michel and Maldonado, Daniel Adrian and Anitescu, Mihai},
  journal={arXiv preprint arXiv:2203.11875},
  year={2022}
}
```


## Table of contents

### Manual

```@contents
Pages = [
    "quickstart/cpu.md",
    "quickstart/cuda.md",
    "man/evaluators.md",
    "man/moi_wrapper.md",
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
