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
        "Evaluators" => "man/evaluators.md",
        "Library" => [
            "Evaluators" => "lib/evaluators.md",
        ]
    ]
)

deploydocs(
    repo = "github.com/exanauts/Argos.jl.git",
    target = "build",
    devbranch = "develop",
    devurl = "dev",
    push_preview = true,
)
