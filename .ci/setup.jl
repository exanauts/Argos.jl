
using Pkg
argos_path = joinpath(@__DIR__, "..")
Pkg.develop(path=argos_path)

Pkg.instantiate()

using CUDA
CUDA.set_runtime_version!(v"11.8")
