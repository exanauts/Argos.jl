module ExaOpt

using LinearAlgebra
using Printf
using ExaPF
using ExaTron
using MathOptInterface

const MOI = MathOptInterface

# Generic type to define new optimization method
abstract type AbstractExaOptimizer end

include("utils.jl")
include("line_model.jl")
include("projected_gradient.jl")
include("conjugategradient.jl")
include("tron_wrapper.jl")
include("activeset.jl")
include("auglag.jl")

end # module
