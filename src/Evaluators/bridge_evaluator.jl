
#=
    BridgeEvaluator
=#
struct BridgeBuffers{VT, MT}
    u::VT
    wx::VT
    wh::VT
    wc::VT
    J::MT
    H::MT
end

"""
    BridgeDeviceEvaluator{Evaluator, DVT, DMT} <: AbstractNLPEvaluator

Bridge an evaluator `nlp` instantiated on the device to use it on
the host memory. The bridge evaluator moves the data between the host
and device automatically.

## Example

```julia-repl

julia> polar = ExaPF.load_polar("case9.m", CUDADevice())

# Load an evaluator on a CUDA GPU
julia> flp = Argos.FullSpaceEvaluator(polar)

julia> bdg = Argos.bridge(flp)

julia> x = Argos.initial(bdg)

julia> @assert isa(x, Array) # x is defined on the host memory

julia> Argos.objective(bdg, x) # evaluate the objective on the device

```

"""
struct BridgeDeviceEvaluator{Evaluator, DVT, DMT} <: AbstractNLPEvaluator
    inner::Evaluator
    buffers::BridgeBuffers{DVT, DMT}
end

function BridgeDeviceEvaluator(nlp::AbstractNLPEvaluator, ::Type{VTD}, ::Type{MTD}) where {VTD, MTD}
    n, m = n_variables(nlp), n_constraints(nlp)
    # Deporting device
    VT = Array{Float64, 1}
    MT = Array{Float64, 2}
    buffers = BridgeBuffers(
        VTD(undef, n),
        VTD(undef, n),
        VTD(undef, n),
        VTD(undef, m),
        MTD(undef, m, n),
        MTD(undef, n, n),
    )
    return BridgeDeviceEvaluator(nlp, buffers)
end

n_variables(nlp::BridgeDeviceEvaluator) = n_variables(nlp.inner)
n_constraints(nlp::BridgeDeviceEvaluator) = n_constraints(nlp.inner)
constraints_type(nlp::BridgeDeviceEvaluator) = constraints_type(nlp.inner)
has_hessian(nlp::BridgeDeviceEvaluator) = has_hessian(nlp.inner)
has_hessian_lagrangian(nlp::BridgeDeviceEvaluator) = has_hessian(nlp.inner)
reset!(nlp::BridgeDeviceEvaluator) = reset!(nlp.inner)

model(nlp::BridgeDeviceEvaluator) = model(nlp.inner)
backend(nlp::BridgeDeviceEvaluator) = backend(nlp.inner)

function bridge(nlp::ReducedSpaceEvaluator{T,VI,VT,MT}) where {T,VI,VT,MT}
    n, m = n_variables(nlp), n_constraints(nlp)
    # Deporting device
    buffers = BridgeBuffers{VT, VT}(
        VT(undef, n),
        VT(undef, n),
        VT(undef, n),
        VT(undef, m),
        VT(undef, m),
        VT(undef, n),
    )
    return BridgeDeviceEvaluator{typeof(nlp), VT, VT}(nlp, buffers)
end
function bridge(nlp::Union{FullSpaceEvaluator{T,VI,VT,MT},StochEvaluator{T,VI,VT,MT}}) where {T,VI,VT,MT}
    n, m = n_variables(nlp), n_constraints(nlp)
    i, j = hessian_structure(nlp)
    nnzh = length(i)
    k, l = jacobian_structure(nlp)
    nnzj = length(k)
    # Deporting device
    buffers = BridgeBuffers{VT,VT}(
        VT(undef, n),
        VT(undef, n),
        VT(undef, n),
        VT(undef, m),
        VT(undef, nnzj),
        VT(undef, nnzh),
    )
    return BridgeDeviceEvaluator{typeof(nlp), VT, VT}(nlp, buffers)
end


function bounds(nlp::BridgeDeviceEvaluator{Ev, DVT, DMT}, attr::AbstractNLPAttribute) where {Ev, DVT, DMT}
    b♭, b♯ = bounds(nlp.inner, attr)
    return b♭ |> Array, b♯ |> Array
end

function initial(nlp::BridgeDeviceEvaluator{Ev, DVT, DMT}) where {Ev, DVT, DMT}
    return initial(nlp.inner) |> Array
end

function update!(nlp::BridgeDeviceEvaluator, u)
    n = length(nlp.buffers.u)
    _copyto!(nlp.buffers.u, 1, u, 1, n)
    return update!(nlp.inner, nlp.buffers.u)
end

objective(nlp::BridgeDeviceEvaluator, u) = objective(nlp.inner, nlp.buffers.u)

function constraint!(nlp::BridgeDeviceEvaluator, cons, u)
    constraint!(nlp.inner, nlp.buffers.wc, nlp.buffers.u)
    _copyto!(cons, 1, nlp.buffers.wc, 1, length(nlp.buffers.wc))
    return
end

function gradient!(nlp::BridgeDeviceEvaluator, grad, u)
    gradient!(nlp.inner, nlp.buffers.wx, nlp.buffers.u)
    _copyto!(grad, 1, nlp.buffers.wx, 1, length(nlp.buffers.wx))
    return
end

function jtprod!(nlp::BridgeDeviceEvaluator, jv, u, v)
    copyto!(nlp.buffers.wc, v)
    jtprod!(nlp.inner, nlp.buffers.wx, nlp.buffers.u, nlp.buffers.wc)
    _copyto!(jv, 1, nlp.buffers.wx, 1, length(u))
    return
end

function jprod!(nlp::BridgeDeviceEvaluator, jv, u, v)
    copyto!(nlp.buffers.wx, v)
    jprod!(nlp.inner, nlp.buffers.wc, nlp.buffers.u, nlp.buffers.wx)
    _copyto!(jv, 1, nlp.buffers.wc, 1, length(jv))
    return
end

function ojtprod!(nlp::BridgeDeviceEvaluator, jv, u, σ, v)
    copyto!(nlp.buffers.wc, v)
    ojtprod!(nlp.inner, nlp.buffers.wx, nlp.buffers.u, σ, nlp.buffers.wc)
    _copyto!(jv, 1, nlp.buffers.wx, 1, length(u))
    return
end

function jacobian!(nlp::BridgeDeviceEvaluator, jac, w)
    jacobian!(nlp.inner, jac, nlp.buffers.u)
    # copyto!(jac, nlp.buffers.J)
    return
end

function jacobian_coo!(nlp::BridgeDeviceEvaluator, jac::AbstractVector, u)
    @assert isa(nlp.buffers.J, AbstractVector)
    jacobian_coo!(nlp.inner, nlp.buffers.J, nlp.buffers.u)
    _copyto!(jac, 1, nlp.buffers.J, 1, length(nlp.buffers.J))
end

function hessprod!(nlp::BridgeDeviceEvaluator, hv, u, v)
    copyto!(nlp.buffers.wx, v)
    hessprod!(nlp.inner, nlp.buffers.wh, nlp.buffers.u, nlp.buffers.g)
    copyto!(hv, nlp.buffers.wh)
    return
end

function hessian!(nlp::BridgeDeviceEvaluator, H, u)
    hessian!(nlp.inner, H, nlp.buffers.u)
    copyto!(H, nlp.buffers.H)
    return
end

function hessian_lagrangian!(nlp::BridgeDeviceEvaluator, H, u, y, σ)
    _copyto!(nlp.buffers.wc, 1, y, 1, length(nlp.buffers.wc))
    hessian_lagrangian!(nlp.inner, H, nlp.buffers.u, nlp.buffers.wc, σ)
    # copyto!(H, nlp.buffers.H)
    return
end

function hessian_lagrangian_coo!(nlp::BridgeDeviceEvaluator, hess, u, y, σ)
    @assert isa(nlp.buffers.H, AbstractVector)
    _copyto!(nlp.buffers.wc, 1, y, 1, length(nlp.buffers.wc))
    hessian_lagrangian_coo!(nlp.inner, nlp.buffers.H, nlp.buffers.u, nlp.buffers.wc, σ)
    _copyto!(hess, 1, nlp.buffers.H, 1, length(nlp.buffers.H))
end

jacobian_structure(nlp::BridgeDeviceEvaluator) = jacobian_structure(nlp.inner)
hessian_structure(nlp::BridgeDeviceEvaluator) = hessian_structure(nlp.inner)
