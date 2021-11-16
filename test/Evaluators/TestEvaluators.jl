module TestEvaluators

@eval Base.Experimental.@optlevel 0

using Test

using FiniteDiff
using LinearAlgebra
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
include("proxal_evaluator.jl")
include("auglag.jl")

function _init(datafile, ::Type{Argos.ReducedSpaceEvaluator}, device)
    return Argos.ReducedSpaceEvaluator(datafile; device=device)
end
function _init(datafile, ::Type{Argos.ProxALEvaluator}, device)
    nlp = Argos.ReducedSpaceEvaluator(datafile; device=device)
    time = Argos.Normal
    return Argos.ProxALEvaluator(nlp, time)
end
function _init(datafile, ::Type{Argos.SlackEvaluator}, device)
    return Argos.SlackEvaluator(datafile; device=device)
end

function runtests(datafile, device, AT)
    @testset "$Evaluator Interface" for Evaluator in [
        Argos.ReducedSpaceEvaluator,
        Argos.AugLagEvaluator,
        Argos.ProxALEvaluator,
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
    @testset "ProxALEvaluator" begin
        nlp = Argos.ReducedSpaceEvaluator(datafile; device=device)
        test_proxal_evaluator(nlp, device, AT)
    end
    @testset "AugLagEvaluator with $Evaluator backend" for Evaluator in [
        Argos.ReducedSpaceEvaluator,
        Argos.ProxALEvaluator,
        Argos.SlackEvaluator,
    ]
        nlp = _init(datafile, Evaluator, device)
        test_auglag_evaluator(nlp, device, AT)
    end
end

end
