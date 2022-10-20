#=
    SCALER
=#
abstract type AbstractScaler end

scale_factor_old(h, tol, η) = max(tol, η / max(1.0, h))
scale_factor(h, tol, η) = min(1.0, η / h)

struct MaxScaler{T, VT} <: AbstractScaler
    scale_obj::T
    scale_cons::VT
    g_min::VT
    g_max::VT
end
function MaxScaler(g_min, g_max)
    @assert length(g_min) == length(g_max)
    m = length(g_min)
    sc = similar(g_min) ; fill!(sc, 1.0)
    return MaxScaler{eltype(g_min), typeof(g_min)}(1.0, sc, g_min, g_max)
end

function MaxScaler(nlp::AbstractNLPEvaluator, u0::VT;
                   η=100.0, tol=1e-8) where VT
    n = n_variables(nlp)
    m = n_constraints(nlp)
    conv = update!(nlp, u0)
    ∇g = similar(u0) ; fill!(∇g, 0)
    gradient!(nlp, ∇g, u0)

    s_obj = scale_factor(ExaPF.xnorm_inf(∇g), tol, η)

    ∇c = VT(undef, n)
    v = VT(undef, m)
    h_s_cons = zeros(m)
    h_v = zeros(m)
    for i in eachindex(h_s_cons)
        fill!(h_v, 0.0)
        h_v[i] = 1.0
        copyto!(v, h_v)
        jtprod!(nlp, ∇c, u0, v)
        h_s_cons[i] = scale_factor(ExaPF.xnorm_inf(∇c), tol, η)
    end
    s_cons = h_s_cons |> VT

    g♭, g♯ = bounds(nlp, Constraints())

    return MaxScaler{typeof(s_obj), typeof(s_cons)}(s_obj, s_cons, s_cons .* g♭, s_cons .* g♯)
end

