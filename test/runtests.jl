
using Test
using LinearAlgebra
using SparseArrays

using KernelAbstractions

using ExaPF
using ExaOpt

using MathOptInterface
const MOI = MathOptInterface

const INSTANCES_DIR = joinpath(dirname(pathof(ExaPF)), "..", "data")
const CASES = ["case9.m", "case30.m"]
ARCHS = Any[(CPU(), Array, SparseMatrixCSC)]

# Load test modules
@isdefined(TestEvaluators) || include("Evaluators/TestEvaluators.jl")

init_time = time()
@testset "Test device specific code on $device" for (device, AT, SMT) in ARCHS
    @info "Test device $device"
    println("Test Evaluators ...")
    tic = time()
    @testset "ExaPF.Evaluator $(case)" for case in CASES
        datafile = joinpath(INSTANCES_DIR, case)
        TestEvaluators.runtests(datafile, device, AT)
    end
    println("Took $(round(time() - tic; digits=1)) seconds.")
end
println()

@testset "Test reduced gradient algorithms on CPU" begin
    @info "Test reduced gradient algorithm ..."
    tic = time()
    include("Algorithms/reduced_gradient.jl")
    include("Algorithms/MOI_wrapper.jl")
    println("Took $(round(time() - tic; digits=1)) seconds.")
end
println("\nTOTAL RUNNING TIME: $(round(time() - init_time; digits=1)) seconds.")

