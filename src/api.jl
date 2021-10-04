
function instantiate_auglag_model(
    datafile;
    scale=true, ρ=10.0, pf_tol=1e-10, line_constraints=false,
    wrap=false, nbatches=1, device=ExaPF.CPU(),
)
    nlp = ReducedSpaceEvaluator(
        datafile;
        device=device, nbatch_hessian=nbatches,
        powerflow_solver=ExaPF.NewtonRaphson(tol=pf_tol),
        line_constraints=line_constraints,
    )
    slk = if wrap
        SlackEvaluator(BridgeDeviceEvaluator(nlp))
    else
        SlackEvaluator(nlp)
    end
    x0 = initial(slk)
    return ExaOpt.AugLagEvaluator(slk, x0; c₀=ρ, scale=scale)
end

