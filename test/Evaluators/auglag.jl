function test_auglag_evaluator(nlp, device, MT)
    u0 = Argos.initial(nlp)
    w♭, w♯ = Argos.bounds(nlp, Argos.Variables())
    # Build penalty evaluator
    @testset "Scaling $scaling" for scaling in [true, false]
        Argos.reset!(nlp)
        pen = Argos.AugLagEvaluator(nlp, u0; scale=scaling)
        VTD, MTD = MT{Float64, 1}, MT{Float64, 2}
        bgd = Argos.BridgeDeviceEvaluator(pen, VTD, MTD)
        u = w♭
        # Update nlp to stay on manifold
        Argos.update!(pen, u)
        # Compute objective
        c = Argos.objective(pen, u)
        c_ref = Argos.inner_objective(pen, u)
        @test isa(c, Real)
        @test c >= c_ref

        ##################################################
        # Update penalty weigth
        # (with a large-enough factor to have a meaningful derivative check)
        ##################################################
        Argos.update_penalty!(pen, η=1e3)
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

        # Test Hessian only on ReducedSpaceEvaluator and SlackEvaluator
        if (
           isa(nlp, Argos.ReducedSpaceEvaluator) ||
           isa(nlp, Argos.SlackEvaluator)
        )
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

