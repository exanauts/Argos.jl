
function test_evaluator_api(nlp, device, M)
    # Test printing
    println(devnull, nlp)

    n = ExaOpt.n_variables(nlp)
    m = ExaOpt.n_constraints(nlp)
    u = ExaOpt.initial(nlp)

    u_min, u_max = ExaOpt.bounds(nlp, ExaOpt.Variables())
    g_min, g_max = ExaOpt.bounds(nlp, ExaOpt.Constraints())
    buffer = get(nlp, ExaPF.PhysicalState())

    # Test consistence
    @test n == length(u)
    @test length(u_min) == length(u_max) == n
    @test myisless(u_min, u_max)
    @test length(g_min) == length(g_max) == m
    if m > 0
        @test myisless(g_min, g_max)
    end

    # Test API
    @test isa(get(nlp, State()), AbstractVector)
    @test isa(get(nlp, ExaOpt.Constraints()), Array{Function})
    @test isa(get(nlp, State()), AbstractVector)
    @test isa(buffer, ExaPF.AbstractBuffer)
    @test ExaOpt.constraints_type(nlp) in [:bound, :equality, :inequality]

    @test isa(ExaOpt.has_hessian(nlp), Bool)
    @test isa(ExaOpt.has_hessian_lagrangian(nlp), Bool)

    # setters
    nbus = get(nlp, PS.NumberOfBuses())
    loads = similar(u, nbus) ; fill!(loads, 1)
    ExaOpt.setvalues!(nlp, PS.ActiveLoad(), loads)
    ExaOpt.setvalues!(nlp, PS.ReactiveLoad(), loads)

    ExaOpt.reset!(nlp)
end

function test_evaluator_callbacks(nlp, device, M; rtol=1e-6)
    # Wrap Evaluator to evaluate FiniteDiff on the CPU
    # (finite_difference_gradient does not support `allowscalar(false)`)
    bdg = ExaOpt.BridgeDeviceEvaluator(nlp, M{Float64, 1}, M{Float64, 2})

    n = ExaOpt.n_variables(nlp)
    m = ExaOpt.n_constraints(nlp)
    u = ExaOpt.initial(nlp)

    u_min, u_max = ExaOpt.bounds(nlp, ExaOpt.Variables())
    g_min, g_max = ExaOpt.bounds(nlp, ExaOpt.Constraints())

    # 1/ update! function
    conv = ExaOpt.update!(nlp, u)
    @test isa(conv, ExaPF.ConvergenceStatus)
    @test conv.has_converged

    # 2/ objective function
    c = ExaOpt.objective(nlp, u)
    @test isa(c, Real)

    # 3/ gradient! function
    function reduced_cost(u_)
        ExaOpt.update!(bdg, u_)
        return ExaOpt.objective(bdg, u_)
    end
    g = similar(u) ; fill!(g, 0)
    ExaOpt.gradient!(nlp, g, u)
    u0 = u |> Array
    grad_fd = FiniteDiff.finite_difference_gradient(reduced_cost, u0)
    @test myisapprox(grad_fd[:], g, rtol=1e-5, atol=1e-5)

    # Constraint
    # 4/ Constraint
    ## Evaluation of the constraints
    if m > 0
        cons = similar(g_min) ; fill!(cons, 0)
        ExaOpt.constraint!(nlp, cons, u)

        ## Evaluation of the transpose-Jacobian product
        jv = similar(u_min) ; fill!(jv, 0.0)
        v = similar(g_min) ; fill!(v, 1.0)
        h_v = v |> Array
        h_cons = cons |> Array
        ExaOpt.jtprod!(nlp, jv, u, v)
        function reduced_cons(u_)
            ExaOpt.update!(bdg, u_)
            ExaOpt.constraint!(bdg, h_cons, u_)
            return dot(h_v, h_cons)
        end
        jv_fd = FiniteDiff.finite_difference_gradient(reduced_cons, u0)

        # TODO: rtol=1e-6 breaks on case30. Investigate why.
        # @test myisapprox(jv, jv_fd[:], rtol=1e-5)

        ## Evaluation of the Jacobian
        J = ExaOpt.jacobian(nlp, u)
        # Test transpose Jacobian vector product
        @test isapprox(jv, J' * v, rtol=rtol)
        # Test Jacobian vector product
        ExaOpt.jprod!(nlp, v, u, jv)
        @test isapprox(J * jv, v)
    end

    ExaOpt.reset!(nlp)
end

function test_evaluator_hessian(nlp, device, M; rtol=1e-6)
    n = ExaOpt.n_variables(nlp)
    @test ExaOpt.has_hessian(nlp)
    function reduced_cost(u_)
        ExaOpt.update!(nlp, u_)
        return ExaOpt.objective(nlp, u_)
    end
    u = ExaOpt.initial(nlp)
    ExaOpt.update!(nlp, u)
    ExaOpt.gradient(nlp, u) # compute the gradient to update the adjoint internally

    # 1/ Hessian-vector product
    hv = similar(u) ; fill!(hv, 0)
    w = similar(u) ; fill!(w, 0)
    h_w = zeros(n) ; h_w[1] = 1.0
    copyto!(w, h_w)
    ExaOpt.hessprod!(nlp, hv, u, w)

    # 2/ Full Hessian
    H = similar(u, n, n) ; fill!(H, 0)
    ExaOpt.hessian!(nlp, H, u)

    # 3/ FiniteDiff
    hess_fd = FiniteDiff.finite_difference_hessian(reduced_cost, u)

    @test H * w == hv
    @test H ≈ hess_fd.data rtol=rtol
end

function test_evaluator_batch_hessian(nlp, device, M; rtol=1e-5)
    n = ExaOpt.n_variables(nlp)
    nbatch = ExaOpt.number_batches_hessian(nlp)
    @test ExaOpt.has_hessian(nlp)
    @test nbatch > 1
    function reduced_cost(u_)
        ExaOpt.update!(nlp, u_)
        return ExaOpt.objective(nlp, u_)
    end

    u = ExaOpt.initial(nlp)
    n = length(u)
    ExaOpt.update!(nlp, u)
    g = ExaOpt.gradient(nlp, u) # compute the gradient to update the adjoint internally

    # 0/ Update Hessian object
    # 1/ Hessian-vector product
    hv = similar(u, n, nbatch) ; fill!(hv, 0)
    w = similar(u, n, nbatch) ; fill!(w, 0)
    w[1, :] .= 1.0
    ExaOpt.hessprod!(nlp, hv, u, w)

    # 2/ Full Hessian
    H = similar(u, n, n) ; fill!(H, 0)
    ExaOpt.hessian!(nlp, H, u)

    # 3/ FiniteDiff
    hess_fd = FiniteDiff.finite_difference_hessian(reduced_cost, u)

    @test H * w ≈ hv
    @test H ≈ hess_fd.data rtol=rtol

    m = ExaOpt.n_constraints(nlp)
    if m > 0
        J = similar(u, m, n)
        ExaOpt.jacobian!(nlp, J, u)
        function reduced_cons(u_)
            cons = similar(u_, m)
            ExaOpt.update!(nlp, u_)
            ExaOpt.constraint!(nlp, cons, u_)
            return cons
        end
        J_fd = FiniteDiff.finite_difference_jacobian(reduced_cons, u)
        @test J ≈ J_fd rtol=rtol
    end
end

