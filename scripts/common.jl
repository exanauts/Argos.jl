
using ExaPF
using LinearAlgebra
using KernelAbstractions
using CUDA
using CUDA.CUSPARSE

using CUDAKernels

const PS = ExaPF.PowerSystem
const LS = ExaPF.LinearSolvers

function build_setup(polar, J, device::CPU, nbatch)
    lufac = lu(J)
    linear_solver = LS.DirectSolver(lufac)
    hessian_lagrangian = if nbatch == 1
        ExaPF.HessianLagrangian(polar, lufac, lufac')
    else
        ExaPF.BatchHessianLagrangian(polar, lufac, lufac', nbatch)
    end
    return (linear_solver, hessian_lagrangian)
end

function build_setup(polar, J, device::CUDADevice, nbatch)
    gJ = CuSparseMatrixCSR(J)
    lufac = CUSOLVERRF.CusolverRfLU(gJ)
    linear_solver = LS.DirectSolver(lufac)
    if nbatch == 1
        blufac = CUSOLVERRF.CusolverRfLU(gJ)
        badjlu = CUSOLVERRF.CusolverRfLU(CuSparseMatrixCSC(J))
        hessian_lagrangian = ExaPF.HessianLagrangian(polar, blufac, badjlu)
    else
        blufac = CUSOLVERRF.CusolverRfLUBatch(gJ, nbatch)
        badjlu = CUSOLVERRF.CusolverRfLUBatch(CuSparseMatrixCSC(J), nbatch)
        hessian_lagrangian = ExaPF.BatchHessianLagrangian(polar, blufac, badjlu, nbatch)
    end
    return (linear_solver, hessian_lagrangian)
end

function build_batch_nlp(datafile, device, nbatch)
    print("Load data\t")
    polar = @time PolarForm(datafile, device)

    constraints = Function[
        ExaPF.voltage_magnitude_constraints,
        ExaPF.active_power_constraints,
        ExaPF.reactive_power_constraints,
    ]
    powerflow_solver = NewtonRaphson(tol=1e-10)

    nbus = get(polar, PS.NumberOfBuses())

    # Take care that J should always have the same sparsity
    # pattern in cusolverRF.
    v0 = polar.network.vbus .+ 0.01 .* rand(ComplexF64, nbus)
    J = ExaPF.matpower_jacobian(polar, State(), ExaPF.power_balance, v0)

    linear_solver, hessian_lagrangian = build_setup(polar, J, device, nbatch)

    nlp = @time ExaPF.ReducedSpaceEvaluator(
        polar; constraints=constraints,
        linear_solver=linear_solver,
        powerflow_solver=powerflow_solver,
        hessian_lagrangian=hessian_lagrangian
    )
    return nlp
end

function build_batch_auglat(datafile, device, nbatch; ρ=0.1, scale=true)
    nlp = build_batch_nlp(datafile, device, nbatch)
    slk = ExaPF.SlackEvaluator(nlp, ExaPF.CPU())
    x0 = ExaPF.initial(slk)
    aug = ExaPF.AugLagEvaluator(slk, x0; c₀=ρ, scale=scale)
    return aug
end

