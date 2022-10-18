# Argos on NVIDIA GPU

Argos has an extension to deport the solving of the OPF
on CUDA GPU. This extension is currently unregistered and has to be installed manually. If you have Argos
installed locally, one can install `ArgosCUDA` using
Julia's package manager:
```
pkg> dev /path/to/Argos.jl/lib/ArgosCUDA.jl

```
Note that `ArgosCUDA` installs nontrivial dependencies, such as [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl/).
For that reason, the installation process takes an additional amount of time.

Then, once `ArgosCUDA` is installed, the procedure is the same as before.

```julia
using Argos, MadNLP
using ArgosCUDA, MadNLPGPU
datafile = joinpath(INSTANCES_DIR, "case118.m")

```

## Full-space method

```julia
ArgosCUDA.run_opf_gpu(datafile, Argos.FullSpace())

```

## Biegler's method (linearize-then-reduce)

```julia
ArgosCUDA.run_opf_gpu(datafile, Argos.BieglerReduction(); linear_solver=LapackGPUSolver)

```

## Dommel & Tinney's method (reduce-then-linearize)

```julia
ArgosCUDA.run_opf_gpu(datafile, Argos.DommelTinney(); linear_solver=LapackGPUSolver)

```
