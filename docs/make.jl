using Firefly
using Documenter

makedocs(;
    modules=[Firefly],
    authors="Miles Lucas <mdlucas@hawaii.edu>",
    repo="https://github.com/mileslucas/Firefly.jl/blob/{commit}{path}#L{line}",
    sitename="Firefly.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://mileslucas.github.io/Firefly.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/mileslucas/Firefly.jl",
)
