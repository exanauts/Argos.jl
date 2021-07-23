
function test_proxal_evaluator(nlp, device, MT)
    u0 = ExaOpt.initial(nlp)
    @testset "ProxALEvaluators ($time)" for time in [ExaOpt.Origin, ExaOpt.Normal, ExaOpt.Final]
        # Build ProxAL evaluator
        prox = ExaOpt.ProxALEvaluator(nlp, time)
        # Wrapper for FiniteDiff
        VTD, MTD = MT{Float64, 1}, MT{Float64, 2}
        bgd = ExaOpt.BridgeDeviceEvaluator(prox, VTD, MTD)

        n = ExaOpt.n_variables(prox)
        w = ExaOpt.initial(prox)
        @test length(w) == n

        # Update nlp to stay on manifold
        conv = ExaOpt.update!(prox, w)
        @test conv.has_converged

        # Compute objective
        c = ExaOpt.objective(prox, w)

        @testset "Gradient & Hessian" begin
            g = similar(w) ; fill!(g, 0)
            ExaOpt.gradient!(prox, g, w)

            # Test evaluation of gradient with Finite Differences
            function reduced_cost(w_)
                ExaOpt.update!(bgd, w_)
                return ExaOpt.objective(bgd, w_)
            end
            grad_fd = FiniteDiff.finite_difference_gradient(reduced_cost, w |> Array)
            @test myisapprox(grad_fd[:], g, rtol=1e-6)

            # Test gradient with non-trivial penalties
            λf = 0.5 .* rand(prox.ng)
            λt = 1.5 .* rand(prox.ng)
            pgf = rand(prox.ng)
            ExaOpt.update_primal!(prox, ExaOpt.Previous(), pgf)
            ExaOpt.update_multipliers!(prox, ExaOpt.Next(), λt)
            ExaOpt.update_multipliers!(prox, ExaOpt.Current(), λf)

            ExaOpt.update!(prox, w)
            fill!(g, 0)
            ExaOpt.gradient!(prox, g, w)
            grad_fd = FiniteDiff.finite_difference_gradient(reduced_cost, w |> Array)
            @test myisapprox(grad_fd[:], g, rtol=1e-6)

            hv = similar(w) ; fill!(hv, 0)
            tgt = similar(w) ; fill!(tgt, 0)
            tgt[1:1] .= 1.0
            ExaOpt.hessprod!(prox, hv, w, tgt)
            H = ExaOpt.hessian(prox, w)

            hess_fd = FiniteDiff.finite_difference_hessian(reduced_cost, w)
            # Take attribute data as hess_fd is of type Symmetric
            @test_broken H ≈ hess_fd.data rtol=1e-6
        end

        @testset "Constraints" begin
            m_I = ExaOpt.n_constraints(prox)
            cons = similar(w, m_I) ; fill!(cons, 0)
            # Evaluate constraints
            ExaOpt.constraint!(prox, cons, w)
            # Transpose Jacobian vector product
            v = similar(w, m_I) ; fill!(v, 0)
            jv = similar(w, n) ; fill!(jv, 0)
            ExaOpt.jtprod!(prox, jv, w, v)

            # Jacobian structure
            if isa(device, CPU)
                rows, cols = ExaOpt.jacobian_structure(prox)
                # Evaluation
                jac = ExaOpt.jacobian(prox, w)
                # Check correctness of transpose Jacobian vector product
                @test jv == jac' * v

                # Jacobian vector product
                ExaOpt.jprod!(prox, v, w, jv)
                @test v == jac * jv
            end
        end

        ExaOpt.reset!(prox)
    end
end
