import Hiop
using LinearAlgebra

# Write GPU kernels
function triu_transfer!(hvec, H)
    n = size(H, 1)

    index = 1
    for i in 1:n, j in 1:n
        @inbounds hvec[index] = 0.5 * (H[i, j] + H[j, i])
        index += 1
    end
end

function build!(nlp)
    @assert Argos.n_constraints(nlp) == 0

    n = Argos.n_variables(nlp)

    hash_x = UInt64(0)

    # Allocate full Hessian
    H = zeros(n, n)

    function _update!(x)
        hx = hash(x)
        if hx != hash_x
            Argos.update!(nlp, x)
            hash_x = hx
        end
    end

    function eval_f(x::Vector{Float64}, prob::HiopProblem)
        _update!(x)
        return Argos.objective(nlp, x)
    end

    function eval_g(x::Vector{Float64}, g::Vector{Float64}, prob::HiopProblem)
        # No constraints!
        return Int32(1)
    end

    function eval_grad_f(x::Vector{Float64}, grad_f::Vector{Float64}, prob::HiopProblem)
        _update!(x)
        Argos.gradient!(nlp, grad_f, x)
        return Int32(1)
    end

    function eval_jac_g(mode::Vector{Symbol}, x::Vector{Float64},
                        iJacS::Vector{Int32}, jJacS::Vector{Int32}, MJacS::Vector{Float64},
                        JacD::Vector{Float64}, prob::HiopProblem)
        # No constraints!
        return Int32(1)
    end

    function eval_h(mode::Vector{Symbol}, x::Vector{Float64}, obj_factor::Float64, lambda::Vector{Float64},
                    iHSS::Vector{Int32}, jHSS::Vector{Int32}, MHSS::Vector{Float64}, HDD::Vector{Float64},
                    iHSD::Vector{Int32}, jHSD::Vector{Int32}, MHSD::Vector{Float64}, prob)
        @assert length(MHSS) == 0
        @assert length(MHSD) == 0
        @assert length(HDD) == n * n

        if :Dense in mode
            _update!(x)
            # Evaluate Hessian
            Argos.hessian!(nlp, H, x)
            HDD .= obj_factor .* H[:]
        end
        return Int32(1)
    end

    # Variables
    ns = 0
    nd = n
    x_L, x_U = Argos.bounds(nlp, Argos.Variables())
    # Constraints
    num_constraints = 0
    g_L = Float64[]
    g_U = Float64[]
    # Hiop (dense) problem
    prob = Hiop.HiopProblem(ns, nd,
        Int32(0), Int32(nd),
        Int32(0), Int32(0),
        Int32(0), Int32(0),
        n, x_L, x_U,
        num_constraints, g_L, g_U,
        eval_f, eval_g, eval_grad_f, eval_jac_g, eval_h
    )
    return prob
end

function Argos.optimize!(prob::Hiop.HiopProblem, nlp, x)
    prob.x0 .= x
    Hiop.solveProblem(prob)

    return (
        status=Hiop.ApplicationReturnStatus[prob.status],
        minimum=prob.obj_val,
        minimizer=prob.x,
    )
end

