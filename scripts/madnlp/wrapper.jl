
using MadNLP

function MadNLP.NonlinearProgram(nlp::ExaOpt.AbstractNLPEvaluator; allocate_buffer=false)

    n = ExaOpt.n_variables(nlp)
    m = ExaOpt.n_constraints(nlp)

    hrows, hcols = ExaOpt.hessian_structure(nlp)
    jrows, jcols = ExaOpt.jacobian_structure(nlp)
    nnz_hess = length(hrows)
    nnz_jac = length(jrows)

    d_x0  = ExaOpt.initial(nlp)
    x0 = d_x0 |> Array
    g  = Vector{Float64}(undef,m)
    xl, xu = ExaOpt.bounds(nlp, ExaOpt.Variables()) .|> Array
    zl = Vector{Float64}(undef,n)
    zu = Vector{Float64}(undef,n)

    l = Vector{Float64}(undef,m)
    gl, gu = ExaOpt.bounds(nlp, ExaOpt.Constraints()) .|> Array

    d_x = similar(d_x0, n)
    d_g = similar(d_x0, n)
    d_c = similar(d_x0, m)

    # Buffers
    H = allocate_buffer ? Matrix{Float64}(undef, n, n) : nothing
    hash_x = UInt64(0)

    function _update!(x::AbstractVector)
        hx = hash(x)
        if hx != hash_x
            hash_x = hx
            copyto!(d_x, x)
            ExaOpt.update!(nlp, d_x)
        end
    end
    function obj(x::AbstractArray{Float64,1})
        _update!(x)
        return ExaOpt.objective(nlp, d_x)
    end
    function obj_grad!(g::AbstractArray{Float64,1},x::AbstractArray{Float64,1})
        _update!(x)
        ExaOpt.gradient!(nlp, d_g, d_x)
        copyto!(g, d_g)
    end
    function con!(c::Array{Float64,1},x::AbstractArray{Float64,1})
        _update!(x)
        ExaOpt.constraint!(nlp, d_c, d_x)
        copyto!(c, d_c)
    end
    function con_jac!(jac::AbstractArray{Float64,1}, x::AbstractArray{Float64,1})
        # _update!(x)
        # J = reshape(jac, m, n)
        # ExaOpt.jacobian!(nlp, J, x)
    end
    function con_jac!(jac::AbstractArray{Float64,2}, x::AbstractArray{Float64,1})
        # _update!(x)
        # ExaOpt.jacobian!(nlp, jac, d_x)
    end
    function lag_hess!(hess::AbstractArray{Float64,2},x::AbstractArray{Float64,1},l::AbstractArray{Float64,1}, sig::Float64)
        _update!(x)
        # Evaluate full reduced Hessian in the preallocated buffer.
        ExaOpt.hessian!(nlp, hess, d_x)
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
    options = Dict{Symbol, Any}(:tol=>1e-5, :max_iter=>30,
                                :linear_solver=>MadNLPLapackGPU)
    ipp = MadNLP.DenseSolver(mnlp, Vector{Float64}, Matrix{Float64};
                             option_dict=options)
    MadNLP.optimize!(ipp)
    return ipp
end

function test_sparse(aug)
    ExaOpt.reset!(aug)
    mnlp = MadNLP.NonlinearProgram(aug; allocate_buffer=true)
    options = Dict{Symbol, Any}(:tol=>1e-5, :max_iter=>30,
                                :linear_solver=>MadNLPLapackCPU)
    ips = MadNLP.Solver(mnlp; option_dict=options)
    MadNLP.optimize!(ips)
    return ips
end

function test_dense_gpu(aug)
    ExaOpt.reset!(aug)
    mnlp = MadNLP.NonlinearProgram(aug)
    options = Dict{Symbol, Any}(:tol=>1e-5, :max_iter=>30,
                                :linear_solver=>MadNLPLapackGPU)
    ipp = MadNLP.DenseSolver(mnlp, CuVector{Float64}, CuMatrix{Float64};
                             option_dict=options)
    # warmstart
    ipp.lag_hess!(ipp.x, ipp.l)
    ipp.cnt.start_time = time()
    MadNLP.optimize!(ipp)
    return ipp
end
