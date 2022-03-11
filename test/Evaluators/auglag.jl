function test_auglag_evaluator(nlp, device, MT)
    u = Argos.initial(nlp)
    w♭, w♯ = Argos.bounds(nlp, Argos.Variables())
    # Build penalty evaluator
    @testset "Scaling $scaling" for scaling in [false, true]
        Argos.reset!(nlp)
        pen = Argos.AugLagEvaluator(nlp, u; scale=scaling)
        VTD, MTD = MT{Float64, 1}, MT{Float64, 2}
        bgd = Argos.BridgeDeviceEvaluator(pen, VTD, MTD)
        # Update nlp to stay on manifold
        Argos.update!(pen, u)
        # Compute objective
        c = Argos.objective(pen, u)
        c_ref = Argos.inner_objective(pen, u)
        @test isa(c, Real)
        @test c >= c_ref

        ##################################################
        # Update penalty weigth
        ##################################################
        Argos.update_penalty!(pen, η=1e0)
        Argos.update_multipliers!(pen)

        ##################################################
        # Callbacks
        ##################################################
        Argos.update!(pen, u)
        obj = Argos.objective(pen, u)
        g = Argos.gradient(pen, u)
        # Compare with finite differences
        function reduced_cost(u_)
            Argos.update!(bgd, u_)
            return Argos.objective(bgd, u_)
        end
        grad_fd = FiniteDiff.finite_difference_gradient(reduced_cost, u |> Array)
        @test myisapprox(grad_fd, g, rtol=1e-4)

        # TODO: we have a nasty side effect
        Argos.reset!(pen)
        n = length(u)
        Argos.update!(pen, u)
        hv = similar(u) ; fill!(hv, 0)
        w = similar(u)
        h_w = zeros(n) ; h_w[1] = 1.0
        copyto!(w, h_w)

        Argos.hessprod!(pen, hv, u, w)
        H = similar(u, n, n) ; fill!(H, 0)
        Argos.hessian!(pen, H, u)
        # Is Hessian vector product relevant?
        @test H * w ≈ hv

        # Is Hessian correct?
        if isa(device, CPU) # test is too slow on GPU...
            hess_fd = FiniteDiff.finite_difference_hessian(reduced_cost, u)
            h_H = H |> Array
            h_H_fd = hess_fd.data |> Array
            @test isapprox(h_H, h_H_fd, rtol=1e-5)
        end

        # Test estimation of multipliers (only on SlackEvaluator)
        if isa(nlp, Argos.SlackEvaluator) && isa(device, CPU)
            λ = Argos.estimate_multipliers(pen, u)
        end
    end
end

