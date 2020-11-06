
function asa(
    nlp,
    u0;
    α_bb=1.0,
    α♭=0.0,
    α♯=1.0,
    tol=1e-4,
    ua_algo=HZ()
)
    status = NotSolved
    u = copy(u0)
    u♭, u♯ = ExaPF.bounds(nlp, ExaPF.Variables())

    solution = ngpa(nlp, u; α♯=α♯, tol=1e-4, active_set=true)
    status = solution.status
    u .= solution.minimizer
    A = solution.active_set

    asa_iter = 1
    while (status != Optimal) && (asa_iter <= 10)
        asa_iter += 1
        println(status)
        println(A)
        if (status == SwitchNGPA)
            solution = ngpa(nlp, u; α♯=α♯, tol=1e-4, active_set=true)
        elseif (status == SwitchCG) || (status == Restart)
            solution = optimize(ua_algo, nlp, u; α♯=α♯, tol=1e-4, active_set=A)
            solution.minimizer .= max.(min.(solution.minimizer, u♯), u♭)
        end
        A = solution.active_set
        u .= solution.minimizer
        println(solution.minimum)
        status = solution.status
    end
    return solution
end

