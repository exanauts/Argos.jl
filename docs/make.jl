using Pkg

using Documenter, ExaOpt

makedocs(
    sitename = "ExaOpt.jl",
    format = Documenter.HTML(
        prettyurls = Base.get(ENV, "CI", nothing) == "true",
        mathengine = Documenter.KaTeX()
    ),
    modules = [ExaOpt],
    repo = "https://github.com/exanauts/ExaOpt.jl/blob/{commit}{path}#{line}",
    strict = true,
    checkdocs = :exports,
    pages = [
        "Home" => "index.md",
        "Evaluators" => "man/evaluators.md",
        "Library" => [
            "Evaluators" => "lib/evaluators.md",
        ]
    ]
)

