#=
    SCALER
=#
abstract type AbstractScaler end

scale_factor(h, tol, η) = max(tol, η / max(1.0, h))
scale_factor2(h, tol, η) = min(1.0, η / h)

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

    s_obj = scale_factor2(ExaPF.xnorm_inf(∇g), tol, η)

    ∇c = VT(undef, n)
    v = VT(undef, m)
    h_s_cons = zeros(m)
    h_v = zeros(m)
    for i in eachindex(h_s_cons)
        fill!(h_v, 0.0)
        h_v[i] = 1.0
        copyto!(v, h_v)
        jtprod!(nlp, ∇c, u0, v)
        h_s_cons[i] = scale_factor2(ExaPF.xnorm_inf(∇c), tol, η)
    end
    s_cons = h_s_cons |> VT

    g♭, g♯ = bounds(nlp, Constraints())

    return MaxScaler{typeof(s_obj), typeof(s_cons)}(s_obj, s_cons, s_cons .* g♭, s_cons .* g♯)
end

function NetworkScaler(nlp::AbstractNLPEvaluator, g_min, g_max; σ=1e-3)
    inner = inner_evaluator(nlp)::ReducedSpaceEvaluator
    model = backend(inner)
    scale_obj = σ
    shift = 0

    m = n_constraints(inner)
    h_scale_cons = zeros(m)
    for cons in inner.constraints
        _m = ExaPF.size_constraint(model, cons)
        if cons === ExaPF.voltage_magnitude_constraints
            μ = 1e-1
        elseif cons === ExaPF.active_power_constraints
            μ = 1e-1
        elseif cons === ExaPF.reactive_power_constraints
            μ = 1e-1
        elseif cons === ExaPF.flow_constraints
            μ = 2e-3
        else
            error("Unsupported constraint")
        end

        h_scale_cons[shift+1:shift+_m] .= μ
        shift += _m
    end

    scale_cons = similar(g_min)
    copyto!(scale_cons, h_scale_cons)

    return MaxScaler{eltype(scale_cons), typeof(scale_cons)}(scale_obj, scale_cons, g_min, g_max)
end
