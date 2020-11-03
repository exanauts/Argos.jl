
using LinearAlgebra

# fix eta k cchoice
# fix quadstep?
#
"""
  HZAW(;)

Implements the approximate Wolfe line search developed in [HZ2005].

theta controls the search in U3 of the paper. It defaults to 0.5, and as
a result uses bisection when (what is U3?)

We tweak the original algorithm slightly, by backtracking into a feasible
region if the original step length results in function values that are not
finite.
"""
struct HZAW{T}
    decrease::T
    curvature::T
    θ::T
    γ::T
end
function HZAW(; decrease=0.1, curvature=0.9, theta=0.5, gamma=2/3)
    if !(0 < decrease ≤ curvature)
        println("Decrease constant must be positive and smaller than the curvature condition.")
    end
    if !(curvature < 1)
        println("Curvature constant must be smaller than one.")
    end
    HZAW(decrease, curvature, theta, gamma)
end

function find_steplength(hzl::HZAW, φ, φ0, dφ0, c, ϵk=1e-6; maxiter=100, ratio=0.1)
    # c = initial(k) but this is done outisde
    # TODO
    T = Float64
    # TODO
    ϵk = T(ϵk)
    δ = T(hzl.decrease)
    σ = T(hzl.curvature)
    ρ = T(5)
    φc = φ(c)
    dφc = grad!(φ, c)
    # Backtrack into feasible region; not part of original algorithm
    ctmp, c = c, c
    iter = 0
    while !isfinite(φc) && iter <= maxiter
        iter += 1
        # don't use interpolation, this is vanilla backtracking
        β = c
        α = ratio*c
        φ_α = φ(α)
        ctmp, c, φc = β, α, φ_α
    end

    # initial convergence
    # Wolfe conditions
    if δ*dφ0 ≥ (φc-φ0)/c && dφc ≥ σ*dφ0
        return c, φc, true
    end
    # Approximate Wolfe conditions
    if (2*δ-1)*dφ0 ≥ dφc ≥ σ*dφ0 && φc ≤ φ0 + ϵk
        return c, φc, true
    end
    # Set up interval
    a0, b0 = bracket(hzl, c, φ, φ0, ϵk, ρ)
    j = 0
    aj, bj  = a0, b0
    # Main loop
    while j < 50 # TODO
        # TODO
        a, b = secant²(hzl, φ, φ0, aj, bj, ϵk)
        if b - a > hzl.γ*(bj - aj)
            c = (a + b)/2
            # TODO
            φc = φ(c)
            dφc = grad!(φ, c)
            a, b = update(hzl, a, b, c, φ, φ0, φc, dφc, ϵk)
        end

        aj, bj = a, b
        j += 1
        if _wolfe(φ0, dφ0, c, φc, dφc, δ, σ, ϵk) || _approx_wolfe(φ0, dφ0, c, φc, dφc, δ, σ, ϵk)
            return c, φc, true
        end
    end
    return T(NaN), T(NaN), false
end
_wolfe(φ0, dφ0, c, φc, dφc, δ, σ, ϵk) =  δ*dφ0 ≥ (φc-φ0)/c && dφc ≥ σ*dφ0
_approx_wolfe(φ0, dφ0, c, φc, dφc, δ, σ, ϵk) = (2*δ-1)*dφ0 ≥ dφc ≥ σ*dφ0 && φc ≤ φ0 + ϵk

"""
   _U3

Used to take step U3 of the updating procedure [HZ, p.123]. The other steps
are in update, but this step is separated out to be able to use it in
step B2 of bracket.
"""
function _U3(hzl::HZAW, φ, φ0, a::T, b::T, c::T, ϵk) where T
    # verified against paper description [p. 123, CG_DESCENT_851]
    # # TODO
    _a, _b = a, c
    # a)
    searching = true
    j = 1
    while searching && j < 50
        # convex combination of _a and _b; 0.5 implies bisection
        d = (1 - hzl.θ)*_a + hzl.θ*_b
        # TODO
        φd, dφd = φ(d)
        if dφd ≥ T(0) # found point of increasing objective; return with upper bound d
            _b = d
            return _a, _b
        else # now dφd < T(0)
            if φd ≤ φ0 + ϵk
                _a = d
            else # φ(d) ≥ φ0 + ϵk
                _b = d
            end
        end
        j += 1
    end
    _a, _b # throw error?
end

function update(hzl::HZ,
                a::T, b::T, c::T,
                φ, φ0, φc, dφc,
                ϵk) where {HZ<:HZAW, T}

    # verified against paper description [p. 123, CG_DESCENT_851]
    #== U0 ==#
    if c ≤ a || c ≥ b # c ∉ (a, b)
        return a, b, (a=false, b=false)
    end
    #== U1 ==#
    if dφc ≥ T(0)
        return a, c, (a=false, b=true)
    else # dφc < T(0)
        #== U2 ==#
        if φc ≤ φ0 + ϵk
            return c, b, (a=true, b=false)
        end
        #== U3 ==#
        a, b = _U3(hzl, φ, a, b, c, ϵk)
        return a, b, (a=a==c, b=b==c)
    end
end

"""
  bracket

Find an interval satisfying the opposite slope condition [OSC] starting from
[0, c] [pp. 123-124, CG_DESCENT_851].
"""
function bracket(hzl::HZAW, c::T, φ, φ0, ϵk, ρ) where T
    # verified against paper description [pp. 123-124, CG_DESCENT_851]
    # Note, we know that dφ(0) < 0 since we're accepted that the current step is in a
    # direction of descent.

    #== B0 ==#
    cj = c
    φcj = φ(cj)
    dφcj = grad!(φ, cj)
    # we only want to store a number, so we don't store all iterates
    ci, φi = T(0), φ0

    maxj = 100
    for j = 1 :maxj
        #==================================================
        B1: φ is increasing at c, set b to cj as this is
        an upper bound, since φ is initially decrea-
        sing.
        ==================================================#
        if dφcj ≥ T(0)
            a, b = ci, cj
            return a, b
        else # dφcj < T(0)
            #== B2 : φ is decreasing at cj but function value is sufficiently larger than
            # φ0, use U3 to update. ==#
            if φcj > φ0 + ϵk
                a, b = _U3(hzl, φ, T(0), cj, c, ϵk)
                return a, b
            end
            #== B3 ==#
            # update ci instead of keeping all c's
            if φcj ≤ φ0 + ϵk
                ci = cj
                φci = φcj
            end
            # expand by factor ρ > 0 (shouldn't this be > 1?)
            cj = ρ*cj
            φcj = φ(cj)
            dφcj = grad!(φ, cj)
        end
    end
end

function secant(hzl::HZAW, a, dφa, b, dφb)
    # verified against paper description [p. 123, CG_DESCENT_851]
    #(a*dφb - b*dφa)/(dφb - dφa)
    # It has been observed that dφa can be very close to dφb,
    # so we avoid taking the difference
    a/(1 - dφa/dφb) + b/(1 - dφb/dφa)
end
function secant²(hzl::HZAW, φ, φ0, a, b, ϵk)
    # verified against paper description [p. 123, CG_DESCENT_851]
    #== S1 ==#
    φa= φ(a)
    dφa  = grad!(φ, a)
    φb = φ(b)
    dφb  = grad!(φ, b)
    c = secant(hzl, a, dφa, b, dφb)

    φc = φ(c)
    dφc  = grad!(φ, c)
    A, B, updates = update(hzl, a, b, c, φ, φ0, φc, dφc, ϵk)
    if updates.b # B == c
        #== S2: c is the upper bound ==#
        φB, dφB = φc, dφc
        _c = secant(hzl, b, dφb, B, dφB)
    elseif updates.a # A == c
        #== S3: c is the lower bound ==#
        φA, dφA = φc, dφc
        _c = secant(hzl, a, dφa, A, dφA)
    end
    updates
    if any(updates)
        #== S4.if: c was upper or lower bound ==#
        φ_c = φ(_c)
        dφ_c  = grad!(φ, _c)
        _a, _b = update(hzl, A, B, _c, φ, φ0, φ_c, dφ_c, ϵk)
        return _a, _b
    else
        #== S4.otherwise: c was neither ==#
        return A, B
    end
end

################################################################################
################################################################################
################################################################################
abstract type CGUpdate end

struct HZ{Tη} <: CGUpdate
    η::Tη # a "forcing term"
end
HZ() = HZ(0.4)
function update_parameter(cg::HZ, d, ∇fz, ∇fx, y, P, P∇fz)
    T = eltype(∇fz)
    θ = T(2)
    η = T(cg.η)
    # βHS = update_parameter(mstyle, HS(), d,  ∇fz, ∇fx, y, P, P∇fz)
    # but let's save the dy calculation from begin repeated for
    # efficiency's sake
    dy = dot(d, y)
    βHS = dot(y, P∇fz)/dy

    # Apply preconditioner to y
    # TODO
    Py = y #apply_preconditioner(mstyle, P, copy(P∇fz), y)
    βN = βHS - θ*dot(y, Py)/dy*dot(d, ∇fz)/dy

    # 2013 version is scale invariant
    # TODO
    Pinv_y = y # apply_inverse_preconditioner(mstyle, P, copy(P∇fz), y)
    ηk = η*dot(d, ∇fx)/dot(d, Pinv_y)
    # 2006 version
    # ηk = -inv(norm(d)*min(T(cg.η), norm(∇fx)))

    βkp1 = max(βN, ηk)
end

struct CGVars{T1, T2, T3}
    y::T1 # change in successive gradients
    d::T2 # search direction
    α::T3
    β::T3
    ls_success::Bool
end

function optimize(algo::HZ, nlp, x0; maxiter=100)
    linesearch = HZAW()
    t0 = time()
    #==============
         Setup
    ==============#
    Tx = eltype(x0)

    objvars = prepare_variables(nlp, x0)
    f0, ∇f0 = objvars.fx, norm(objvars.∇fx, Inf) # use user norm

    y, d, α, β = copy(objvars.∇fz), -copy(objvars.∇fx), Tx(0), Tx(0)
    cgvars = CGVars(y, d, α, β, true)

    k = 1
    objvars, P, cgvars = iterate(algo, cgvars, objvars, nlp, linesearch)
    is_converged = norm(∇f0, Inf) <= 1e-5

    while k < maxiter && !any(is_converged)
        k += 1
        objvars, P, cgvars = iterate(algo, cgvars, objvars, nlp, linesearch; is_first=false)
        is_converged = norm(objvars.∇fx, Inf) <= 1e-5
        println(norm(objvars.∇fx, Inf))
    end
    x, fx, ∇fx, z, fz, ∇fz, B = objvars
    # return ConvergenceInfo(approach, (beta=β, ρs=norm(x.-z), ρx=norm(x), minimizer=z, fx=fx, minimum=fz, ∇fx=∇fx, ∇fz=∇fz, f0=f0, ∇f0=∇f0, iter=k, time=time()-t0), options)
end

function iterate(cg::HZ, cgvars::CGVars, objvars, nlp, linesearch; is_first=true)
    # split up the approach into the hessian approximation scheme and line search
    x, fx, ∇fx, z, fz, ∇fz, B, Pg = objvars
    Tx = eltype(x)

    y, d, α, β = cgvars.y, cgvars.d, cgvars.α, cgvars.β

    # Move nexts into currs
    fx = fz
    copyto!(x, z)
    copyto!(∇fx, ∇fz)

    # Precondition current gradient and calculate the search direction
    P∇fz = ∇fz
    @. d = -P∇fz + β*d

    # TODO
    # φ = _lineobjective(mstyle, problem, ∇fz, z, x, d, fx, dot(∇fx, d))
    φ = LineModel(nlp, x, d, ∇fz, copy(z))
    α_0 = 1e-3 #initial(nlp, α, x, φ, fx, dot(d, ∇fx), ∇fx, is_first)

    # Perform line search along d
    α, f_α, ls_success = find_steplength(linesearch, φ, fx, dot(d, ∇fx), Tx(α_0), 1e-6)

    # Calculate final step vector and update the state
    if ls_success
        @. z = z + α * d
        ExaPF.update!(nlp, z)
        fz = ExaPF.objective(nlp, z)
        ExaPF.gradient!(nlp, ∇fz, z)
        @. y = ∇fz - ∇fx
    else
        # if no succesful search direction is found, reset to gradient
        y .= .-∇fz
    end
    β = update_parameter(cg, d, ∇fz, ∇fx, y, I, P∇fz)

    return (x=x, fx=fx, ∇fx=∇fx, z=z, fz=fz, ∇fz=∇fz, B=nothing, Pg=Pg), I, CGVars(y, d, α, β, ls_success)
end

function initial(nlp, α, x, φ, φ₀, dφ₀, ∇fx, is_first)
    T = eltype(x)
    ψ₀ = T(0.01)
    ψ₁ = T(0.1)
    ψ₂ = T(2.0)
    quadstep = true
    if is_first isa Nothing
        if !all(x .≈ T(0)) # should we define "how approx we want?"
            return ψ₀ * norm(x, Inf)/norm(∇fx, Inf)
        elseif !(φ₀ ≈ T(0))
            return ψ₀ * abs(φ₀)/norm(∇fx, 2)^2
        else
            return T(1)
        end
    elseif quadstep
        R = ψ₁*α
        # TODO
        φR = φ(R)
        if φR ≤ φ₀
            c = (φR - φ₀ - dφ₀*R)/R^2
            if c > 0
               return -dφ₀/(T(2)*c) # > 0 by df0 < 0 and c > 0
            end
        end
    end
    return ψ₂*α
end

function prepare_variables(nlp, x0)
    z = x0
    x = copy(z)
    ∇fz = copy(x0)
    # TODO
    ExaPF.update!(nlp, x0)
    fz = ExaPF.objective(nlp, x0)
    ExaPF.gradient!(nlp, ∇fz, x0)

    fx = copy(fz)
    ∇fx = copy(∇fz)

    Pg = ∇fz
    return (x=x, fx=fx, ∇fx=∇fx, z=z, fz=fz, ∇fz=∇fz, B=nothing, Pg=Pg)
end

