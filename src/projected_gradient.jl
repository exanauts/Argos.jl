
function projected_gradient(
    nlp,
    uk;
    max_iter=1000,
    ls_itermax=30,
    α0=1.0,
    β=0.4,
    τ=1e-4,
    tol=1e-5,
    bfgs=false,
    verbose_it=100,
)

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
        ExaPF.update!(nlp, uk)

        # evaluate cost
        c = ExaPF.objective(nlp, uk)
        # Evaluate cost of problem without penalties
        c_ref = nlp.scaler.scale_obj * ExaPF.objective(nlp.inner, uk)
        ExaPF.gradient!(nlp, grad, uk)

        # compute control step
        # Armijo line-search (Bertsekas, 1976)
        dk = H * grad
        step = α0
        for j_ls in 1:ls_itermax
            step *= β
            ExaPF.project!(wk, uk .- step .* dk, u♭, u♯)
            ExaPF.update!(nlp, wk)
            ft = ExaPF.objective(nlp, wk)
            if ft <= c - τ * dot(dk, wk .- uk)
                break
            end
        end

        # step = αi
        wk .= uk .- step * dk
        ExaPF.project!(uk, wk, u♭, u♯)

        # Stopping criteration: uₖ₊₁ - uₖ
        ## Dual infeasibility
        norm_grad = norm(uk .- u_prev, Inf)
        ## Primal infeasibility
        inf_pr = ExaPF.primal_infeasibility(nlp.inner, nlp.cons)

        # check convergence
        if (i % verbose_it == 0)
            @printf("%6d %.6e %.3e %.2e %.2e %.2e\n", i, c, c - c_ref, norm_grad, inf_pr, step)
        end

        if bfgs
            push!(H, uk .- u_prev, grad .- grad_prev)
            grad_prev .= grad
        end
        u_prev .= uk
        # Check whether we have converged nicely
        if (norm_grad < tol)
            converged = true
            break
        end
    end
    return uk, norm_grad, n_iter
end


# Non-monotone gradient projection algorithm
function ngpa(
    nlp,
    uk;
    max_iter=1000,
    ls_itermax=30,
    α_bb=1.0,
    α♭=0.0,
    α♯=1.0,
    β=0.4,
    δ=1e-4,
    tol=1e-5,
    verbose_it=100,
    ls_algo=3,
)

    u_prev = copy(uk)
    grad = copy(uk)
    wk = copy(uk)
    u♭ = nlp.inner.u_min
    u♯ = nlp.inner.u_max

    # Initial evaluation
    ExaPF.update!(nlp, uk)
    f = ExaPF.objective(nlp, uk)
    ExaPF.gradient!(nlp, grad, uk)
    # Memory
    grad_prev = copy(grad)

    norm_grad = norm(grad, Inf)
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

    n_up = 0
    for i in 1:max_iter
        n_iter += 1

        ExaPF.project!(wk, uk .- α_bb .* grad, u♭, u♯)
        # Feasible direction
        dk = wk .- uk

        ##################################################
        # Armijo line-search
        step = 1.0
        d∇g = dot(dk, grad)
        for j_ls in 1:ls_itermax
            ExaPF.project!(wk, uk .+ step .* dk, u♭, u♯)
            conv = ExaPF.update!(nlp, wk)
            ft = ExaPF.objective(nlp, wk)
            if ft <= min(fᵣ, f♯_ref) + step * δ * d∇g
                break
            end
            step *= β
            # Step introduced in Birgin & Martinez & Raydan (2012)
            α = - 0.5 * step^2 * d∇g / (ft - f - step * d∇g)
            if σ1_arm * step <= α <= σ2_arm * step
                step = α
            else
                step *= β
            end
        end

        uk .= wk
        # Objective
        f = ExaPF.objective(nlp, uk)
        c_ref = ExaPF.inner_objective(nlp, uk)
        # Gradient
        ExaPF.gradient!(nlp, grad, uk)

        # Stopping criteration: uₖ₊₁ - uₖ
        ## Dual infeasibility
        norm_grad = norm(uk .- u_prev, Inf)
        ## Primal infeasibility
        inf_pr = ExaPF.primal_infeasibility(nlp.inner, nlp.cons ./ nlp.scaler.scale_cons)

        # check convergence
        if (i % verbose_it == 0)
            @printf("%6d %.6e %.3e %.2e %.2e %.2e\n", i, f, f - c_ref, norm_grad, inf_pr, step)
        end

        ##################################################
        ## Update parameters
        sk = uk - u_prev
        yk = grad - grad_prev

        ##################################################
        ## Update Barzilai-Borwein step
        flag_bb = 0
        if !isnothing(findfirst(0.0 .< abs.(dk) .< α_bb .* abs.(grad)))
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
                    α_bb = max(t, step)
                    j_bb = 0
                end
            else
                # Everything went ok. Set new step to Barzilai-Borwein value
                α_bb = dot(sk, sk) / dot(sk, yk)
                j_bb = 0
            end
        end
        α_bb = min(α♯, α_bb)
        # Update history
        u_prev .= uk
        grad_prev .= grad

        ##################################################
        ## Update reference value
        # Update maximum value
        buffer_costs[i % M_ref + 1] = f
        f♯_ref = maximum(buffer_costs)
        if ls_algo == 1
            w = .2
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

        # Check whether we have converged nicely
        if (norm_grad < tol)
            converged = true
            break
        end
    end
    return uk, norm_grad, n_iter
end

