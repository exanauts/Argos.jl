module ArgosCUDA

using LinearAlgebra
using SparseArrays

using CUDA
using CUDA.CUSPARSE
using CUSOLVERRF

import KernelAbstractions as KA
import KernelAbstractions: @kernel, @index

using ExaPF
const LS = ExaPF.LinearSolvers

using Argos
using MadNLP

include("kernels.jl")
include("sparse.jl")
include("reduction.jl")
include("api.jl")

end # module
