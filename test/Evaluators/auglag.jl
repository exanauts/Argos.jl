function test_auglag_evaluator(nlp, device, MT)
    u0 = ExaOpt.initial(nlp)
    w♭, w♯ = ExaOpt.bounds(nlp, ExaOpt.Variables())
    # Build penalty evaluator
    @testset "Scaling $scaling" for scaling in [true, false]
        ExaOpt.reset!(nlp)
        pen = ExaOpt.AugLagEvaluator(nlp, u0; scale=scaling)
        bgd = ExaOpt.BridgeDeviceEvaluator(pen, CPU())
        u = w♭
        # Update nlp to stay on manifold
        ExaOpt.update!(pen, u)
        # Compute objective
        c = ExaOpt.objective(pen, u)
        c_ref = ExaOpt.inner_objective(pen, u)
        @test isa(c, Real)
        @test c >= c_ref
        inf_pr2 = ExaOpt.primal_infeasibility(pen, u)
        @test inf_pr2 >= 0.0

        ##################################################
        # Update penalty weigth
        # (with a large-enough factor to have a meaningful derivative check)
        ##################################################
        ExaOpt.update_penalty!(pen, η=1e3)
        ExaOpt.update_multipliers!(pen)

        ##################################################
        # Callbacks
        ##################################################
        ExaOpt.update!(pen, u)
        obj = ExaOpt.objective(pen, u)
        g = ExaOpt.gradient(pen, u)
        # Compare with finite differences
        function reduced_cost(u_)
            ExaOpt.update!(bgd, u_)
            return ExaOpt.objective(bgd, u_)
        end
        grad_fd = FiniteDiff.finite_difference_gradient(reduced_cost, u |> Array)
        @test myisapprox(grad_fd, g, rtol=1e-5)

        # Test Hessian only on ReducedSpaceEvaluator and SlackEvaluator
        if (
           isa(nlp, ExaOpt.ReducedSpaceEvaluator) ||
           isa(nlp, ExaOpt.SlackEvaluator)
        )
            n = length(u)
            ExaOpt.update!(pen, u)
            hv = similar(u) ; fill!(hv, 0)
            w = similar(u)
            h_w = zeros(n) ; h_w[1] = 1.0
            copyto!(w, h_w)

            ExaOpt.hessprod!(pen, hv, u, w)
            H = similar(u, n, n) ; fill!(H, 0)
            ExaOpt.hessian!(pen, H, u)
            # Is Hessian vector product relevant?
            @test H * w ≈ hv
            # Is Hessian correct?
            hess_fd = FiniteDiff.finite_difference_hessian(reduced_cost, u)

            h_H = H |> Array
            h_H_fd = hess_fd.data |> Array

            @test isapprox(h_H, h_H_fd, rtol=1e-5)
        end
        # Test estimation of multipliers (only on SlackEvaluator)
        if isa(nlp, ExaOpt.SlackEvaluator) && isa(device, CPU)
            λ = ExaOpt.estimate_multipliers(pen, u)
        end
    end
end

