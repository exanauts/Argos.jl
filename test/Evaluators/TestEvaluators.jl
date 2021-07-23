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
using ExaOpt

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

function _init(datafile, ::Type{ExaOpt.ReducedSpaceEvaluator}, device)
    constraints = Function[
        ExaPF.voltage_magnitude_constraints,
        ExaPF.active_power_constraints,
        ExaPF.reactive_power_constraints,
        # ExaPF.flow_constraints,
    ]
    return ExaOpt.ReducedSpaceEvaluator(datafile; device=device, constraints=constraints)
end
function _init(datafile, ::Type{ExaOpt.ProxALEvaluator}, device)
    nlp = ExaOpt.ReducedSpaceEvaluator(datafile; device=device)
    time = ExaOpt.Normal
    return ExaOpt.ProxALEvaluator(nlp, time)
end
function _init(datafile, ::Type{ExaOpt.SlackEvaluator}, device)
    return ExaOpt.SlackEvaluator(datafile; device=device)
end

function runtests(datafile, device, AT)
    @testset "$Evaluator Interface" for Evaluator in [
        ExaOpt.ReducedSpaceEvaluator,
        ExaOpt.AugLagEvaluator,
        ExaOpt.ProxALEvaluator,
        ExaOpt.SlackEvaluator,
        ExaOpt.FeasibilityEvaluator,
    ]
        nlp = Evaluator(datafile; device=device)
        test_evaluator_api(nlp, device, AT)
        test_evaluator_callbacks(nlp, device, AT)
    end
    @testset "$Evaluator Hessian" for Evaluator in [
        ExaOpt.ReducedSpaceEvaluator,
        ExaOpt.AugLagEvaluator,
    ]
        nlp = Evaluator(datafile; device=device)
        test_evaluator_hessian(nlp, device, AT)
    end
    @testset "ReducedSpaceEvaluator BatchHessian" begin
        nlp = ExaOpt.ReducedSpaceEvaluator(datafile; device=device, nbatch_hessian=2)
        test_evaluator_batch_hessian(nlp, device, AT)
    end
    @testset "ProxALEvaluator" begin
        nlp = ExaOpt.ReducedSpaceEvaluator(datafile; device=device)
        test_proxal_evaluator(nlp, device, AT)
    end
    @testset "AugLagEvaluator with $Evaluator backend" for Evaluator in [
        ExaOpt.ReducedSpaceEvaluator,
        ExaOpt.ProxALEvaluator,
        ExaOpt.SlackEvaluator,
    ]
        nlp = _init(datafile, Evaluator, device)
        test_auglag_evaluator(nlp, device, AT)
    end
end

end
