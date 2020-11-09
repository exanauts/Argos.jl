## Line-search procedure
struct LineModel
    model::ExaPF.AbstractNLPEvaluator
    u::AbstractVector
    d::AbstractVector
    g::AbstractVector
    ut::AbstractVector
end

function _update!(ϕ::LineModel, α)
    ϕ.ut .= ϕ.u .+ α .* ϕ.d
    u♭, u♯ = bounds(ϕ.model, ExaPF.Variables())
    project!(ϕ.ut, ϕ.ut, u♭, u♯)
    ExaPF.update!(ϕ.model, ϕ.ut)
end

function (ϕ::LineModel)(α)
    _update!(ϕ, α)
    return ExaPF.objective(ϕ.model, ϕ.ut)
end

function grad!(ϕ::LineModel, α)
    active_gradient!(ϕ.model, ϕ.g, ϕ.ut)
    return dot(ϕ.g, ϕ.d)
end

function max_step(ϕ::LineModel)
    u♭, u♯ = ExaPF.bounds(ϕ.model, ExaPF.Variables())
    d, u = ϕ.d, ϕ.u
    step♭, step♯ = Inf, 0
    bdist = Inf

    for i in eachindex(ϕ.u)
        if d[i] > 0.0
            t = u♯[i] - u[i]
            bdist = min(t, bdist)
            step = t / d[i]
            step♭, step♯ = min(step♭, step), max(step♯, step)
            bdist = min(u[i] - u♭[i], bdist)
        elseif d[i] < 0.0
            t = u[i] - u♭[i]
            bdist = min(t, bdist)
            step = -t / d[i]
            step♭, step♯ = min(step♭, step), max(step♯, step)
            bdist = min(u♯[i] - u[i], bdist)
        else
            bdist = min(u[i] - u♭[i] , u♯[i] - u[i], bdist)
        end
    end

    return (bdist , step♭, step♯)
end
