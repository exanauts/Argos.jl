
using MadNLP

function MadNLP.NonlinearProgram(nlp::ExaOpt.AbstractNLPEvaluator)

    n = ExaOpt.n_variables(nlp)
    m = ExaOpt.n_constraints(nlp)

    hrows, hcols = ExaOpt.hessian_structure(nlp)
    jrows, jcols = ExaOpt.jacobian_structure(nlp)
    nnz_hess = length(hrows)
    nnz_jac = length(jrows)

    x0  = ExaOpt.initial(nlp)
    g  = Vector{Float64}(undef,m)
    xl, xu = ExaOpt.bounds(aug, ExaOpt.Variables())
    zl = Vector{Float64}(undef,n)
    zu = Vector{Float64}(undef,n)

    l = Vector{Float64}(undef,m)
    gl, gu = ExaOpt.bounds(aug, ExaOpt.Constraints())

    # Buffers
    H = Matrix{Float64}(undef, n, n)
    hash_x = UInt64(0)

    function _update!(x::AbstractVector)
        hx = hash(x)
        if hx != hash_x
            ExaOpt.update!(nlp, x)
            hash_x = hx
        end
    end
    function obj(x::AbstractArray{Float64,1})
        _update!(x)
        return ExaOpt.objective(nlp, x)
    end
    function obj_grad!(f::AbstractArray{Float64,1},x::AbstractArray{Float64,1})
        _update!(x)
        ExaOpt.gradient!(nlp, f, x)
    end
    function con!(c::Array{Float64,1},x::AbstractArray{Float64,1})
        _update!(x)
        ExaOpt.constraint!(nlp, c, x)
    end
    function con_jac!(jac::AbstractArray{Float64,1}, x::AbstractArray{Float64,1})
        _update!(x)
        J = reshape(jac, m, n)
        ExaOpt.jacobian!(nlp, J, x)
    end
    function con_jac!(jac::AbstractArray{Float64,2}, x::AbstractArray{Float64,1})
        _update!(x)
        ExaOpt.jacobian!(nlp, jac, x)
    end
    function lag_hess!(hess::AbstractArray{Float64,2},x::AbstractArray{Float64,1},l::AbstractArray{Float64,1}, sig::Float64)
        _update!(x)
        # Evaluate full reduced Hessian in the preallocated buffer.
        ExaOpt.hessian!(nlp, hess, x)
        hess .*= sig
        return
    end
    function lag_hess!(hess::AbstractArray{Float64,1},x::AbstractArray{Float64,1},l::AbstractArray{Float64,1}, sig::Float64)
        _update!(x)
        # Evaluate full reduced Hessian in the preallocated buffer.
        ExaOpt.hessian!(nlp, H, x)
        # Only dense Hessian supported now
        index = 1
        @inbounds for i in 1:n, j in 1:i
            # Hessian is symmetric, and MOI considers only the lower
            # triangular part. We average the values from the lower
            # and upper triangles for stability.
            hess[index] = 0.5 * sig * (H[i, j] + H[j, i])
            index += 1
        end
    end
    function hess_sparsity!(I,J)
        copy!(I, hrows)
        copy!(J, hcols)
        return
    end
    function jac_sparsity!(I,J)
        copy!(I, jrows)
        copy!(J, jcols)
        return
    end

    # Build MadNLP model
    return MadNLP.NonlinearProgram(
        n, m, nnz_hess, nnz_jac, 0., x0, g, l, zl, zu, xl, xu, gl, gu,
        obj, obj_grad!, con!, con_jac!, lag_hess!, hess_sparsity!, jac_sparsity!,
        MadNLP.INITIAL,
        Dict{Symbol, Any}()
    )
end

function ExaOpt.optimize!(
    opt::MadNLP.NonlinearProgram, nlp::ExaOpt.AbstractNLPEvaluator, x0;
    options=Dict{Symbol, Any}()
)
    opt.x .= x0
    ips = MadNLP.Solver(opt;option_dict=copy(options))
    MadNLP.optimize!(ips)
    return
end

function test_dense(aug)
    ExaOpt.reset!(aug)
    mnlp = MadNLP.NonlinearProgram(aug)
    options = Dict{Symbol, Any}(:tol=>1e-5, :max_iter=>30)
    ipp = MadNLP.DenseSolver(mnlp, Vector{Float64}, Matrix{Float64};
                             option_dict=options)
    MadNLP.optimize!(ipp)
    return ipp
end

function test_sparse(aug)
    ExaOpt.reset!(aug)
    mnlp = MadNLP.NonlinearProgram(aug)
    options = Dict{Symbol, Any}(:tol=>1e-5, :max_iter=>30)
    ips = MadNLP.Solver(mnlp; option_dict=options)
    MadNLP.optimize!(ips)
    return ips
end

