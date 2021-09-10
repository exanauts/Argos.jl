
include(joinpath(@__DIR__, "..", "hessian", "checkhess.jl"))

rel_gap(x, y) = abs.(x .- y) ./ max.(1.0, abs.(y))

# datafile = PROBLEMS["case118"]

INSTANTIATE = true

if INSTANTIATE
    aug_nolines = @time build_problem(datafile; line_constraints=false)
    aug_nolines.ρ = 0.0
    aug_nolines.λ .= 10.0
    aug_lines = @time build_problem(datafile; line_constraints=true)
    aug_lines.ρ = 0.0
    aug_lines.λ .= 10.0
    nlp = ExaOpt.inner_evaluator(aug_nolines)
end

N = min(50, ExaOpt.n_variables(nlp))

n_nl = ExaOpt.n_variables(aug_nolines)

u_nl = ExaOpt.initial(aug_nolines)
h1_nl, h2_nl = check_hessian_fd_hessian(aug_nolines, u_nl, N)

u_wl = ExaOpt.initial(aug_lines)
h1_wl, h2_wl = check_hessian_fd_hessian(aug_lines, u_wl, N)

rel_nl = rel_gap(h1_nl, h2_nl)
rel_wl = rel_gap(h1_wl, h2_wl)

ind_nl = findall(rel_nl .> 1e-1)
ind_wl = findall(rel_wl .> 1e-1)
