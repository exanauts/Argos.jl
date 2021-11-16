# Verify solutions against matpower results
using Test
using ExaPF
using FiniteDiff
using LinearAlgebra
using KernelAbstractions

@testset "RGM Optimal Power flow 9 bus case" begin
    datafile = joinpath(dirname(pathof(ExaPF)), "..", "data", "case9.m")

    nlp = Argos.ReducedSpaceEvaluator(datafile)
    uk = Argos.initial(nlp)

    # solve power flow
    Argos.update!(nlp, uk)

    # reduced gradient method
    iterations = 0
    iter_max = 100
    step = 0.0001
    norm_grad = 10000
    norm_tol = 1e-5

    iter = 1

    # initial gradient
    grad = similar(uk)
    wk = similar(uk)
    up = copy(uk)
    fill!(grad, 0)

    while norm_grad > norm_tol && iter < iter_max
        Argos.update!(nlp, uk)
        c = Argos.objective(nlp, uk)
        Argos.gradient!(nlp, grad, uk)
        # compute control step
        wk = uk - step*grad
        Argos.project!(uk, wk, nlp.u_min, nlp.u_max)
        norm_grad = norm(uk .- up, Inf)
        iter += 1
        up .= uk
    end
    @test iter == 39
    @test isapprox(uk, [1.1, 1.1, 1.1, 1.343109921105559, 0.9421135274454701], atol=1e-4)
end

