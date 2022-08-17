# Generic type to define new optimization method
abstract type AbstractExaOptimizer end

include("common.jl")
include("auglag.jl")
