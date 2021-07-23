
function projected_gradient(
    nlp,
    uk;
    max_iter=1000,
    ls_itermax=30,
    α0=1.0,
    β=0.4,
    τ=1e-4,
    tol=1e-5,
    verbose_it=Inf,
)

    f = Inf
    status = NotSolved
    u_prev = copy(uk)
    grad = copy(uk)
    wk = copy(uk)
    u♭ = nlp.inner.u_min
    u♯ = nlp.inner.u_max
    H = I
    norm_grad = Inf
    n_iter = 0

    for i in 1:max_iter
        n_iter += 1
        # solve power flow and compute gradients
        update!(nlp, uk)

        # evaluate cost
        c = objective(nlp, uk)
        # Evaluate cost of problem without penalties
        c_ref = nlp.scaler.scale_obj * objective(nlp.inner, uk)
        gradient!(nlp, grad, uk)

        # compute control step
        # Armijo line-search (Bertsekas, 1976)
        dk = H * grad
        step = α0
        for j_ls in 1:ls_itermax
            step *= β
            project!(wk, uk .- step .* dk, u♭, u♯)
            update!(nlp, wk)
            ft = objective(nlp, wk)
            if ft <= c - τ * dot(dk, wk .- uk)
                break
            end
        end

        # step = αi
        wk .= uk .- step * dk
        project!(uk, wk, u♭, u♯)
        f = objective(nlp, uk)

        # Stopping criteration: uₖ₊₁ - uₖ
        ## Dual infeasibility set to norm of Cauchy step
        #
        # sₖ(α) = P[xₖ - α ∇f(xₖ)] - xₖ
        #
        norm_grad = norm(uk .- u_prev, Inf)
        ## Primal infeasibility
        inf_pr = primal_infeasibility(nlp.inner, nlp.cons)

        # check convergence
        if (i % verbose_it == 0)
            @printf("%6d %.6e %.3e %.2e %.2e %.2e\n", i, c, c - c_ref, norm_grad, inf_pr, step)
        end

        u_prev .= uk
        # Check whether we have converged nicely
        if (norm_grad < tol)
            break
        end
    end

    return solution = (
        status=status,
        minimum=f,
        minimizer=uk,
        iter=n_iter,
        inf_du=norm_grad,
    )
end


# Non-monotone gradient projection algorithm
function ngpa(
    nlp,
    uk;
    active_set=false,
    max_iter=1000,
    ls_itermax=20,
    α_bb=1.0,
    α♭=0.0,
    α♯=1.0,
    β=0.4,
    δ=1e-4,
    tol=1e-4,
    verbose_it=Inf,
    ls_algo=3,
)

    n = n_variables(nlp)
    # Status
    status = NotSolved
    u_prev = copy(uk)
    ∇f = copy(uk)
    wk = copy(uk)
    dk = copy(uk)
    u♭, u♯ = bounds(nlp, Variables())

    # Initial evaluation
    update!(nlp, uk)
    f = objective(nlp, uk)
    gradient!(nlp, ∇f, uk)

    # Memory
    grad_prev = copy(∇f)
    # Active gradient
    grad_act = copy(∇f)

    norm_grad = norm(∇f, Inf)
    n_iter = 0
    ## Line-search params
    j_bb = 0
    flag_bb = 1
    θ_bb = 0.975
    m_bb = 10
    ## Reference function params
    L_ref = 3
    M_ref = 8
    P_ref = 40
    γ1_ref = M_ref / L_ref
    γ2_ref = P_ref / M_ref
    l_ref = 0
    p_ref = 0
    f♭_ref = f
    f♯_ref = f
    fc_ref = f
    buffer_costs = Float64[f for i in 1:M_ref]

    η_ref = 0.8
    Q_ref = 1.0
    fᵣ = f
    # Armijo params
    σ1_arm = 0.1
    σ2_arm = 0.9

    # Active set
    U  = Int[]
    A  = Int[]
    A_hash_1 = hash(A)
    A_hash_2 = hash(A)
    μ_act = 0.1
    ρ_act = 0.5

    n_up = 0
    while (n_iter <= max_iter) && (status == NotSolved)
        n_iter += 1

        feasible_direction!(dk, wk, uk, ∇f, α_bb, u♭, u♯)

        ##################################################
        # Armijo line-search
        step = 1.0
        d∇g = dot(dk, ∇f)
        for j_ls in 1:ls_itermax
            project_step!(wk, uk, dk, u♭, u♯, step)
            conv = update!(nlp, wk)
            f₊ = objective(nlp, wk)
            if f₊ <= min(fᵣ, f♯_ref) + step * δ * d∇g
                break
            end
            # Step introduced in Birgin & Martinez & Raydan (2012)
            α = - 0.5 * step^2 * d∇g / (f₊ - f - step * d∇g)
            if σ1_arm * step <= α <= σ2_arm * step
                step = α
            else
                step *= β
            end
        end

        ##################################################
        ## Update parameters
        uk .= wk
        # Objective
        f = objective(nlp, uk)
        # Gradient
        gradient!(nlp, ∇f, uk)
        sk = uk - u_prev
        yk = ∇f - grad_prev

        # Stopping criteration: uₖ₊₁ - uₖ
        ## Dual infeasibility
        feasible_direction!(dk, wk, uk, ∇f, 1.0, u♭, u♯)
        norm_grad = norm(dk, Inf)
        ## Dual infeasibility set to norm of Cauchy step
        #
        # sₖ(α) = P[xₖ - α ∇f(xₖ)] - xₖ
        #
        # norm_grad = norm(uk .- u_prev, Inf)

        # check convergence
        if (n_iter % verbose_it == 0)
            inf_pr = primal_infeasibility(nlp.inner, nlp.cons ./ nlp.scaler.scale_cons)
            @printf("%6d %.6e %.2e %.2e %.2e\n", n_iter, f, norm_grad, inf_pr, step)
        end

        ##################################################
        ## Update Barzilai-Borwein step
        flag_bb = 0
        if !isnothing(findfirst(0.0 .< abs.(dk) .< α_bb .* abs.(∇f)))
            flag_bb = 1
        end
        if step == 1.0
            j_bb += 1
        else
            flag_bb = 1
        end
        θ = dot(sk, yk) / (norm(sk) * norm(yk))
        if (j_bb >= m_bb) || (θ >= θ_bb) || (flag_bb == 1)
            # Non-convexity detected
            if dot(sk, yk) <= 0.0
                if j_bb >= 1.5 * m_bb
                    t_bb = min(norm(uk, Inf), 1) / norm(dk, Inf)
                    α_bb = max(t_bb, step)
                    j_bb = 0
                end
            else
                # Everything went ok. Set new step to Barzilai-Borwein value
                α_bb = dot(sk, sk) / dot(sk, yk)
                j_bb = 0
            end
        end
        α_bb = max(min(α♯, α_bb), α♭)

        ##################################################
        # Update history
        u_prev .= uk
        grad_prev .= ∇f

        ##################################################
        ## Update reference value
        # Update maximum value
        buffer_costs[n_iter % M_ref + 1] = f
        f♯_ref = maximum(buffer_costs)
        if ls_algo == 1
            w = 0.15
            fᵣ = w * f♯_ref + (1 - w) * f
        elseif ls_algo == 2
            qt = η_ref * Q_ref + 1.0
            fᵣ = (η_ref * Q_ref * fᵣ + f) / qt
            Q_ref = qt
        elseif ls_algo == 3
            # Update parameters
            p_ref = (step == 1.0) ? p_ref + 1 : 0
            if f < f♭_ref
                fc_ref = f
                f♭_ref = f
                l_ref = 0
            else
                l_ref += 1
            end
            fc_ref = (f > fc_ref) ? f : fc_ref

            # Step 1
            if l_ref == L_ref
                ratio = (f♯_ref - f♭_ref) / (fc_ref - f♭_ref)
                fᵣ = (ratio > γ1_ref) ? fc_ref : f♯_ref
            end
            if p_ref > P_ref
                ratio = (fᵣ - f) / (f♯_ref - f)
                fᵣ = (f♯_ref > f) && (ratio >= γ2_ref) ? f♯_ref : fᵣ
            end
        end

        # Active-set embedding
        if active_set && (n_iter >= 5)
            grad_act .= ∇f
            ExaOpt.active!(grad_act, uk, u♭, u♯)
            feasible_direction!(dk, wk, uk, ∇f, 1.0, u♭, u♯)
            ndk = norm(dk, Inf)
            # Update active set
            active_set!(A, uk, u♭, u♯)
            undecided_set!(U, ∇f, dk, uk, u♭, u♯)

            if isempty(U)
                if norm(grad_act, Inf) < μ_act * ndk
                    μ_act = ρ_act * μ_act
                else
                    status = SwitchCG
                end
            elseif A_hash_1 == A_hash_2 == hash(A)
                if norm(grad_act, Inf) >= μ_act * ndk
                    status = SwitchCG
                end
            end
            A_hash_1 = A_hash_2
            A_hash_2 = hash(A)
        end

        # Check whether we have converged nicely
        if (norm_grad < tol)
            status = Optimal
        end
    end

    if n_iter >= max_iter
        status = MaxIterations
    end

    return (
        status=status,
        minimum=f,
        minimizer=uk,
        iter=n_iter,
        inf_du=norm_grad,
        active_set=A,
    )
end

