
function asa(
    nlp,
    u0;
    max_iter=100,
    α_bb=1.0,
    α♭=0.0,
    α♯=1.0,
    tol=1e-5,
    ua_algo=HZ(),
    verbose_it=1,
)
    status = NotSolved
    u = copy(u0)
    u♭, u♯ = ExaPF.bounds(nlp, ExaPF.Variables())

    solution = ngpa(nlp, u; α♯=α♯, α_bb=α♯, tol=tol, active_set=true)
    status = solution.status
    u .= solution.minimizer
    A = solution.active_set

    asa_iter = 1
    while (status != Optimal) && (asa_iter <= max_iter)
        asa_iter += 1
        if (status == SwitchNGPA)
            solution = ngpa(nlp, u; α♯=α♯, α_bb=α♯, tol=tol, active_set=true)
        elseif (status == SwitchCG) || (status == Restart) || (status == MaxIterations)
            solution = optimize(ua_algo, nlp, u; α♯=α♯, tol=tol, active_set=A, maxiter=1_000)
            # solution.minimizer .= max.(min.(solution.minimizer, u♯), u♭)
        end
        A = solution.active_set
        u .= solution.minimizer
        println(solution.minimum)
        status = solution.status
    end
    return solution
end

