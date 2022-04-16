module TestEvaluators

@eval Base.Experimental.@optlevel 0

using Test

using FiniteDiff
using LinearAlgebra
using DelimitedFiles
using LazyArtifacts
using Random
using SparseArrays
using KernelAbstractions

using ExaPF
import ExaPF: PowerSystem, LinearSolvers
using Argos

const PS = PowerSystem
const LS = LinearSolvers

function myisless(a, b)
    h_a = a |> Array
    h_b = b |> Array
    return h_a <= h_b
end

function myisapprox(a, b; options...)
    h_a = a |> Array
    h_b = b |> Array
    return isapprox(h_a, h_b; options...)
end

include("api.jl")
include("auglag.jl")

function _init(datafile, ::Type{Argos.ReducedSpaceEvaluator}, device)
    return Argos.ReducedSpaceEvaluator(datafile; device=device, auglag=true)
end
# function _init(datafile, ::Type{Argos.ProxALEvaluator}, device)
#     nlp = Argos.ReducedSpaceEvaluator(datafile; device=device)
#     time = Argos.Normal
#     return Argos.ProxALEvaluator(nlp, time)
# end
function _init(datafile, ::Type{Argos.SlackEvaluator}, device)
    return Argos.SlackEvaluator(datafile; device=device, auglag=true)
end

function runtests(datafile, device, AT)
    @testset "$Evaluator Interface" for Evaluator in [
        Argos.ReducedSpaceEvaluator,
        Argos.AugLagEvaluator,
        Argos.SlackEvaluator,
        Argos.FeasibilityEvaluator,
    ]
        nlp = Evaluator(datafile; device=device)
        test_evaluator_api(nlp, device, AT)
        test_evaluator_callbacks(nlp, device, AT)
    end
    @testset "$Evaluator Hessian" for Evaluator in [
        Argos.ReducedSpaceEvaluator,
        Argos.AugLagEvaluator,
    ]
        nlp = Evaluator(datafile; device=device)
        test_evaluator_hessian(nlp, device, AT)
    end
    @testset "ReducedSpaceEvaluator BatchHessian" begin
        nlp = Argos.ReducedSpaceEvaluator(datafile; device=device, nbatch_hessian=2)
        test_evaluator_batch_hessian(nlp, device, AT)
    end
    @testset "AugLagEvaluator with $Evaluator backend" for Evaluator in [
        Argos.ReducedSpaceEvaluator,
        Argos.SlackEvaluator,
    ]
        nlp = _init(datafile, Evaluator, device)
        test_auglag_evaluator(nlp, device, AT)
    end
    if isa(device, CPU) # Currently supported only on the CPU
        @testset "Argos.FullSpaceEvaluator Interface" begin
            nlp = Argos.FullSpaceEvaluator(datafile)
            test_evaluator_api(nlp, device, AT)
            test_evaluator_callbacks(nlp, device, AT)
            test_evaluator_hessian_lagrangian(nlp, device, AT)
            test_evaluator_sparse_callbacks(nlp, device, AT)
        end
        @testset "Argos.StochEvaluator Interface" begin
            nblocks = 5
            case_name = split(split(datafile, '/')[end], '.')[1]
            demands = joinpath(artifact"ExaData", "ExaData", "mp_demand")
            pload = readdlm(joinpath(demands, "$(case_name)_oneweek_168.Pd"))[:, 1:nblocks] ./ 100
            qload = readdlm(joinpath(demands, "$(case_name)_oneweek_168.Qd"))[:, 1:nblocks] ./ 100

            stoch = Argos.StochEvaluator(datafile, pload, qload)
            test_evaluator_api(stoch, device, AT)
            test_evaluator_callbacks(stoch, device, AT)
            test_evaluator_hessian_lagrangian(stoch, device, AT)
            test_evaluator_sparse_callbacks(stoch, device, AT)
        end
    end
end

end
