module ArgosCUDA

using LinearAlgebra
using SparseArrays

using CUDA
using CUDA.CUSPARSE
using CUSOLVERRF

using KernelAbstractions
using CUDAKernels

using ExaPF
const LS = ExaPF.LinearSolvers

using Argos
using MadNLP

include("kernels.jl")
include("sparse.jl")
include("reduction.jl")

end # module
