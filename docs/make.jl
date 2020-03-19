using Firefly
using Documenter

setup = quote
    using Firefly
    using Random
    Random.seed!(42)
end

DocMeta.setdocmeta!(Firefly, :DocTestSetup, setup; recursive = true)

makedocs(;
    modules = [Firefly],
    authors = "Miles Lucas <mdlucas@hawaii.edu>",
    repo = "https://github.com/mileslucas/Firefly.jl/blob/{commit}{path}#L{line}",
    sitename = "Firefly.jl",
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://mileslucas.com/Firefly.jl",
        assets = String[],
    ),
    pages = [
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo = "github.com/mileslucas/Firefly.jl",
    push_preview = true
)
