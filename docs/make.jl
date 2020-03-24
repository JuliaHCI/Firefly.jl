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
    repo = "https://github.com/juliahci/Firefly.jl/blob/{commit}{path}#L{line}",
    sitename = "Firefly.jl",
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://juliahci.github.io/Firefly.jl",
        assets = String[],
    ),
    pages = [
        "Home" => "index.md",
    ],
    strict = true
)

deploydocs(;
    repo = "github.com/JuliaHCI/Firefly.jl",
    push_preview = true
)
