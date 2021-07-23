# Generic type to define new optimization method
abstract type AbstractExaOptimizer end

include("common.jl")
include("projected_gradient.jl")
include("conjugategradient.jl")
include("activeset.jl")
include("auglag.jl")
