
function asa(
    nlp,
    u0;
    max_iter=100,
    α_bb=1.0,
    α♭=0.0,
    α♯=1.0,
    cg_tol=1e-5,
    gp_tol=1e-4,
    ua_algo=HZ(),
    verbose_it=1,
)
    status = NotSolved
    u = copy(u0)
    u♭, u♯ = ExaPF.bounds(nlp, ExaPF.Variables())

    solution = ngpa(nlp, u; α♯=α♯, α_bb=α♯, tol=gp_tol, active_set=true)
    status = solution.status
    u .= solution.minimizer
    A = solution.active_set

    asa_iter = 1
    while (status != Optimal) && (asa_iter <= max_iter)
        asa_iter += 1
        if (status == SwitchNGPA)
            solution = ngpa(nlp, u; α♯=α♯, α_bb=α♯, tol=gp_tol, active_set=true)
        elseif (status == SwitchCG) || (status == Restart) || (status == MaxIterations)
            solution = optimize(ua_algo, nlp, u; α♯=α♯, tol=cg_tol, active_set=A, maxiter=1_000)
            # solution.minimizer .= max.(min.(solution.minimizer, u♯), u♭)
        end
        A = solution.active_set
        u .= solution.minimizer
        status = solution.status
        println(solution.minimum, "\t", length(A), "\t", status, "\t", solution.inf_du)
    end
    asa_solution = (
        status=status,
        minimum=solution.minimum,
        minimizer=solution.minimizer,
        iter=asa_iter,
        inf_du=solution.inf_du,
        active_set=A,
    )
    return asa_solution
end

