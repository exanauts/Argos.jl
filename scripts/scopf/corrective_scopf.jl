
using Revise
using Random
using NLPModels
using ExaPF
using Argos
using MadNLP
using MadNLPHSL

function generate_loads(model, nscen, magnitude)
    nbus = get(model, ExaPF.PS.NumberOfBuses())
    stack = ExaPF.NetworkStack(model)
    pload_det = stack.pload |> Array
    qload_det = stack.qload |> Array

    # pload_det .*= 2.12
    # qload_det .*= 2.12

    has_load = (pload_det .> 0)

    Random.seed!(1)
    pload = magnitude .* (randn(nbus, nscen) .* has_load) .+ pload_det
    qload = magnitude .* (randn(nbus, nscen) .* has_load) .+ qload_det
    return pload, qload
end

function instantiate!(blk::Argos.OPFModel)
    x0 = NLPModels.get_x0(blk)
    nnzj = NLPModels.get_nnzj(blk)
    jac = zeros(nnzj)
    NLPModels.jac_coord!(blk, x0, jac)
    return
end

function solve_corrective_opf(case, nscen)
    model = ExaPF.PolarForm(case)
    pl, ql = generate_loads(model, nscen, 0.05)
    ev = Argos.CorrectiveEvaluator(model, pl, ql; line_constraints=true, epsilon=1e-4)
    blk = Argos.OPFModel(ev)
    instantiate!(blk)
    solver = MadNLP.MadNLPSolver(
        blk;
        dual_initialized=true,
        linear_solver=Ma27Solver,
        max_iter=250,
        print_level=MadNLP.DEBUG,
        tol=1e-5,
    )
    MadNLP.solve!(solver)
    return (
        solver=solver,
        model=ev,
    )
end

function solve_corrective_tracking_opf(case, nscen, stack_ref)
    model = ExaPF.PolarForm(case)
    pl, ql = generate_loads(model, nscen, 0.05)
    ev = Argos.CorrectiveEvaluator(model, pl, ql; line_constraints=true, tracking=true, stack_ref=stack_ref)
    blk = Argos.OPFModel(ev)
    instantiate!(blk)
    solver = MadNLP.MadNLPSolver(
        blk;
        dual_initialized=true,
        linear_solver=Ma27Solver,
        max_iter=250,
        print_level=MadNLP.DEBUG,
        tol=1e-5,
    )
    MadNLP.solve!(solver)
    return (
        solver=solver,
        model=ev,
    )
end

function analyse_solution(sol)
    nbus = get(Argos.model(sol.model), ExaPF.PS.NumberOfBuses())
    ngen = get(Argos.model(sol.model), ExaPF.PS.NumberOfGenerators())
    nx = ExaPF.number(Argos.model(sol.model), State())

    y = sol.solver.y
    active_cons = [1:nx*nscen; nx*nscen .+ findall(abs.(y[1+nx*nscen:end]) .> 1e-5)]

    # Jacobian
    J = sol.model.jac.J[active_cons, :]
    Gx = sol.model.jac.J[1:nx*nscen, 1:nx*nscen]
    Gxu = J[1:nx*nscen, :]

    sJ = svd(Array(J)).S
    sgx = svd(Array(Gx)).S

    println()
    println("* Min singular value Gx: ", sgx[end])
    println("* Min singular value J:  ", sJ[end])
end


nscen = 1
mode = :corrective

case = "../matpower/data/case9.m"
sol = if mode == :corrective
    solve_corrective_opf(case, nscen)
elseif mode == :tracking
    # For this example, we took as a reference MATPOWER's base solution
    model = ExaPF.PolarForm(case)
    stack_ref = ExaPF.NetworkStack(model)
    solve_corrective_tracking_opf(case, nscen, stack_ref)
end

# Fetch solution
stack = sol.model.stack
ngen = get(Argos.model(sol.model), ExaPF.PS.NumberOfGenerators())
# Adjustment
delta = stack.vuser[1:nscen]
# Reference setpoint for generators
pg_ref = stack.vuser[nscen+1:nscen+ngen]

