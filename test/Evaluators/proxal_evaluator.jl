
function test_proxal_evaluator(nlp, device, MT)
    u0 = Argos.initial(nlp)
    @testset "ProxALEvaluators ($time)" for time in [Argos.Origin, Argos.Normal, Argos.Final]
        # Build ProxAL evaluator
        prox = Argos.ProxALEvaluator(nlp, time)
        # Wrapper for FiniteDiff
        VTD, MTD = MT{Float64, 1}, MT{Float64, 2}
        bgd = Argos.BridgeDeviceEvaluator(prox, VTD, MTD)

        n = Argos.n_variables(prox)
        w = Argos.initial(prox)
        @test length(w) == n

        # Update nlp to stay on manifold
        conv = Argos.update!(prox, w)
        @test conv.has_converged

        # Compute objective
        c = Argos.objective(prox, w)

        @testset "Gradient & Hessian" begin
            g = similar(w) ; fill!(g, 0)
            Argos.gradient!(prox, g, w)

            # Test evaluation of gradient with Finite Differences
            function reduced_cost(w_)
                Argos.update!(bgd, w_)
                return Argos.objective(bgd, w_)
            end
            grad_fd = FiniteDiff.finite_difference_gradient(reduced_cost, w |> Array)
            @test myisapprox(grad_fd[:], g, rtol=1e-6)

            # Test gradient with non-trivial penalties
            λf = 0.5 .* rand(prox.ng)
            λt = 1.5 .* rand(prox.ng)
            pgf = rand(prox.ng)
            Argos.update_primal!(prox, Argos.Previous(), pgf)
            Argos.update_multipliers!(prox, Argos.Next(), λt)
            Argos.update_multipliers!(prox, Argos.Current(), λf)

            Argos.update!(prox, w)
            fill!(g, 0)
            Argos.gradient!(prox, g, w)
            grad_fd = FiniteDiff.finite_difference_gradient(reduced_cost, w |> Array)
            @test myisapprox(grad_fd[:], g, rtol=1e-6)

            hv = similar(w) ; fill!(hv, 0)
            tgt = similar(w) ; fill!(tgt, 0)
            tgt[1:1] .= 1.0
            Argos.hessprod!(prox, hv, w, tgt)
            H = Argos.hessian(prox, w)

            hess_fd = FiniteDiff.finite_difference_hessian(reduced_cost, w)
            # Take attribute data as hess_fd is of type Symmetric
            @test_broken H ≈ hess_fd.data rtol=1e-6
        end

        @testset "Constraints" begin
            m_I = Argos.n_constraints(prox)
            cons = similar(w, m_I) ; fill!(cons, 0)
            # Evaluate constraints
            Argos.constraint!(prox, cons, w)
            # Transpose Jacobian vector product
            v = similar(w, m_I) ; fill!(v, 0)
            jv = similar(w, n) ; fill!(jv, 0)
            Argos.jtprod!(prox, jv, w, v)

            # Jacobian structure
            if isa(device, CPU)
                rows, cols = Argos.jacobian_structure(prox)
                # Evaluation
                jac = Argos.jacobian(prox, w)
                # Check correctness of transpose Jacobian vector product
                @test jv == jac' * v

                # Jacobian vector product
                Argos.jprod!(prox, v, w, jv)
                @test v == jac * jv
            end
        end

        Argos.reset!(prox)
    end
end
