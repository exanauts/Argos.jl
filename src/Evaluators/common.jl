# Common interface for AbstractNLPEvaluator
#
function Base.show(io::IO, nlp::AbstractNLPEvaluator)
    n = n_variables(nlp)
    m = n_constraints(nlp)
    println(io, "A $(typeof(nlp)) object")
    println(io, "    * #vars: ", n)
    println(io, "    * #cons: ", m)
end

## Generic callbacks
function constraint(nlp::AbstractNLPEvaluator, x)
    cons = similar(x, n_constraints(nlp)) ; fill!(cons, 0)
    constraint!(nlp, cons, x)
    return cons
end

function gradient(nlp::AbstractNLPEvaluator, x)
    grad = similar(x) ; fill!(grad, 0)
    gradient!(nlp, grad, x)
    return grad
end

function jacobian(nlp::AbstractNLPEvaluator, x)
    n = n_variables(nlp)
    m = n_constraints(nlp)
    J = similar(x, m, n) ; fill!(J, 0)
    jacobian!(nlp, J, x)
    return J
end

# Joint Objective Jacobian transpose vector product (default implementation)
function ojtprod!(nlp::AbstractNLPEvaluator, jv, u, σ, v)
    gradient!(nlp, jv, u)
    jv .*= σ  # scale gradient
    jtprod!(nlp, jv, u, v)
    return
end

# Generate Hessian using Hessian-vector product routine
macro define_hessian(function_name, target_function, args...)
    fname = Symbol(function_name)
    argstup = Tuple(args)
    quote
        function $(esc(fname))(nlp::AbstractNLPEvaluator, hess, $(map(esc, argstup)...))
            @assert has_hessian(nlp)
            n = n_variables(nlp)
            v = similar(x)
            @inbounds for i in 1:n
                hv = @view hess[:, i]
                fill!(v, 0)
                v[i:i] .= 1.0
                $target_function(nlp, hv, $(map(esc, argstup)...), v)
            end
        end
    end
end

@define_hessian hessian! hessprod! x
@define_hessian hessian_lagrangian! hessian_lagrangian_prod! x y σ
@define_hessian hessian_lagrangian_penalty! hessian_lagrangian_penalty_prod! x y σ D
@define_hessian jacobian! jprod! x

function hessian(nlp::AbstractNLPEvaluator, x)
    n = n_variables(nlp)
    H = similar(x, n, n) ; fill!(H, 0.0)
    hessian!(nlp, H, x)
    return H
end

function set_batch_tangents!(seeds, offset, n, n_batches)
    @assert size(seeds) == (n, n_batches)
    fill!(seeds, 0.0)
    @inbounds for j in 1:n_batches
        seeds[j+offset, j] = 1.0
    end
end

# Sparse callbacks

# By default, sparse callbacks assume that the structure is dense.
function jacobian_structure(nlp::AbstractNLPEvaluator)
    m, n = n_constraints(nlp), n_variables(nlp)
    nnzj = m * n
    rows, cols = zeros(Int, nnzj), zeros(Int, nnzj)
    idx = 1
    for i in 1:n # number of variables
        for c in 1:m #number of constraints
            rows[idx] = c ; cols[idx] = i
            idx += 1
        end
    end
    return rows, cols
end

function hessian_structure(nlp::AbstractNLPEvaluator)
    n = n_variables(nlp)
    # Return lower-triangular matrix
    rows = Int[r for r in 1:n for c in 1:r]
    cols = Int[c for r in 1:n for c in 1:r]
    return rows, cols
end

function jacobian_coo!(nlp::AbstractNLPEvaluator, jac::AbstractVector, x)
    n, m = n_variables(nlp), n_constraints(nlp)
    @assert length(jac) == n * m
    J = reshape(jac, m, n) # convert as dense matrix
    jacobian!(nlp, J, x)
end

function hessian_lagrangian_coo!(nlp::AbstractNLPEvaluator, hess, x, μ, σ)
    n = n_variables(nlp)
    # Evaluate full reduced Hessian in the preallocated buffer.
    H = get_hessian_buffer(nlp)::Array{Float64, 2}
    hessian_lagrangian!(nlp, H, x, μ, σ)
    # Only dense Hessian supported now
    index = 1
    @inbounds for i in 1:n, j in 1:i
        # Hessian is symmetric, and MOI considers only the lower
        # triangular part. We average the values from the lower
        # and upper triangles for stability.
        hess[index] = 0.5 * (H[i, j] + H[j, i])
        index += 1
    end
end

function tril_mapping(H::SparseMatrixCSC)
    n, m = size(H)
    csc2tril = Int[]
    k = 1
    @inbounds for j in 1:m
        for c in H.colptr[j]:H.colptr[j+1]-1
            i = H.rowval[c]
            if j <= i
                push!(csc2tril, k)
            end
            k += 1
        end
    end
    return csc2tril
end

function transfer2tril!(hessvals::AbstractVector, H::SparseMatrixCSC, csc2tril)
    Hz = nonzeros(H)
    hessvals .= Hz[csc2tril]
end

