
using Pkg
Pkg.instantiate()

using CUDA

argos_path = joinpath(@__DIR__, "..")
Pkg.develop(path=argos_path)

CUDA.set_runtime_version!(v"11.8")
