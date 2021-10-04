module ExaOpt

using LinearAlgebra
using Printf
using SparseArrays

using ExaPF
import ExaPF: LinearSolvers, AutoDiff, PowerSystem
const LS = LinearSolvers
const PS = PowerSystem

import MathOptInterface
const MOI = MathOptInterface

using MadNLP

include("Evaluators/Evaluators.jl")

# Wrappers
include("Wrappers/MOI_wrapper.jl")
include("Wrappers/MadNLP_wrapper.jl")

# Algorithms
include("Algorithms/Algorithms.jl")

# Exposed function
include("api.jl")

end # module
