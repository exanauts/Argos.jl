using Pkg

Pkg.develop(PackageSpec(path=joinpath(dirname(@__FILE__), "..")))
# when first running instantiate
Pkg.instantiate()

using Documenter, Argos

makedocs(
    sitename = "Argos.jl",
    format = Documenter.HTML(
        prettyurls = Base.get(ENV, "CI", nothing) == "true",
        mathengine = Documenter.KaTeX()
    ),
    modules = [Argos],
    repo = "https://github.com/exanauts/Argos.jl/blob/{commit}{path}#{line}",
    # strict = true,
    checkdocs = :exports,
    pages = [
        "Home" => "index.md",
        "Quickstart" => [
            "quickstart/cpu.md",
            "quickstart/cuda.md",
        ],
        "Manual" => [
            "Overview" => "man/overview.md",
            "Full-space" => "man/fullspace.md",
            "Reduced-space" => "man/reducedspace.md",
            "Evaluators" => "man/evaluators.md",
        ],
        "Wrappers" => [
            "MathOptInterface" => "man/moi_wrapper.md",
            "NLPModels" => "man/nlpmodel_wrapper.md",
        ],
        "Library" => [
            "Evaluators API" => "lib/api.md",
            "Evaluators Type" => "lib/evaluators.md",
        ],
        "References" => "references.md",
    ]
)

deploydocs(
    repo = "github.com/exanauts/Argos.jl.git",
    target = "build",
    devbranch = "develop",
    devurl = "dev",
    push_preview = true,
)
