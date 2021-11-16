using Test
using Profile
using FiniteDiff

function check_hessian_fd_hessian(aug, u, n_)
    n = length(u)

    Argos.update!(aug, u)
    g = similar(u)
    Argos.gradient!(aug, g, u)

    H = similar(u, n, n) ; fill!(H, 0)
    @time Argos.hessian!(aug, H, u)

    uu = copy(u)
    function reduced_cost(u_)
        uu[1:n_] .= u_
        Argos.update!(aug, uu)
        return Argos.objective(aug, uu)
    end

    hess_fd = FiniteDiff.finite_difference_hessian(reduced_cost, u[1:n_])
    return (H[1:n_, 1:n_], hess_fd)
end

function check_hessian_fd_jacobian(aug, u, n_)
    n = length(u)

    Argos.update!(aug, u)
    g = similar(u)
    Argos.gradient!(aug, g, u)
    H = similar(u, n, n) ; fill!(H, 0)
    @time Argos.hessian!(aug, H, u)

    uu = copy(u)
    gg = similar(u) ; fill!(gg, 0)
    function reduced_gradient(u_)
        uu[1:n_] .= u_
        Argos.update!(aug, uu)
        Argos.gradient!(aug, gg, uu)
        return gg[1:n_]
    end

    hess_fd = FiniteDiff.finite_difference_jacobian(reduced_gradient, u[1:n_])
    return (H[1:n_, 1:n_], hess_fd)
end

function test_hessian(aug; nfd=-1)
    n = Argos.n_variables(aug)
    n_ = (nfd == -1) ? n : nfd
    H, hess_fd = check_hessian(aug, n_)
    Base.@info(1, "h", H)
    Base.@info(1, "h", hess_fd)
    @test isapprox(H, hess_fd, rtol=1e-5)
end

