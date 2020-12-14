
function tron_solve(nlp, x0; options=Dict())
    block_data = MOI.NLPBlockData(nlp)
    x_L, x_U = ExaPF.bounds(nlp, ExaPF.Variables())
    tron_solve(block_data.evaluator, x0, x_L, x_U; options=options)
end
function tron_solve(evaluator, x0, x_L, x_U; options=Dict())
    features = MOI.features_available(evaluator)
    @assert (:Grad in features)
    @assert (:Hess in features)

    MOI.initialize(evaluator, [:Grad, :Hess])
    hessian_sparsity = MOI.hessian_lagrangian_structure(evaluator)

    # Objective callback
    objective_scale = 1.0

    eval_f_cb(x) = objective_scale * MOI.eval_objective(evaluator, x)

    function eval_grad_f_cb(x, grad_f)
        MOI.eval_objective_gradient(evaluator, grad_f, x)
        rmul!(grad_f, objective_scale)
    end

    function eval_h_cb(x, mode, rows, cols, obj_factor, lambda, values)
        if mode == :Structure
            for i in 1:length(hessian_sparsity)
                rows[i] = hessian_sparsity[i][1]
                cols[i] = hessian_sparsity[i][2]
            end
        else
            obj_factor *= objective_scale
            MOI.eval_hessian_lagrangian(evaluator, values, x, obj_factor, lambda)
        end
    end

    start_time = time()

    # If Tron problem was not created, create it.
    nnzH = length(hessian_sparsity)

    inner = createProblem(length(x_L), x_L, x_U,
                          nnzH,
                          eval_f_cb, eval_grad_f_cb, eval_h_cb)

    inner.x .= x0

    for (name, value) in options
        setOption(inner, name, value)
    end

    solveProblem(inner)
    solution = (
        status=inner.status,
        minimizer=inner.x,
        inf_du=inner.gnorm_inf,
        iter=inner.minor_iter,
    )

    return solution
end
