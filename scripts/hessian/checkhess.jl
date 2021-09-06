using Test
using Profile
using FiniteDiff

function check_hessian(aug, n_)
    ExaOpt.reset!(aug.inner)
    u = ExaOpt.initial(aug)
    n = length(u)

    ExaOpt.update!(aug, u)
    g = ExaOpt.gradient(aug, u)

    H = similar(u, n, n) ; fill!(H, 0)
    @time ExaOpt.hessian!(aug, H, u)
    @time ExaOpt.hessian!(aug, H, u)

    uu = copy(u)

    function reduced_cost(u_)
        uu[1:n_] .= u_
        ExaOpt.update!(aug, uu)
        return ExaOpt.objective(aug, uu)
    end

    # Is Hessian correct?
    hess_fd = FiniteDiff.finite_difference_hessian(reduced_cost, u[1:n_])
    return (H[1:n_, 1:n_], hess_fd)
end

function test_hessian(aug; nfd=-1)
    n = ExaOpt.n_variables(aug)
    n_ = (nfd == -1) ? n : nfd
    H, hess_fd = check_hessian(aug, n_)
    Base.@info(1, "h", H)
    Base.@info(1, "h", hess_fd)
    @test isapprox(H, hess_fd, rtol=1e-5)
end

