using KalmanFilterEngine
using Documenter

DocMeta.setdocmeta!(KalmanFilterEngine, :DocTestSetup, :(using KalmanFilterEngine); recursive=true)

makedocs(;
    modules=[KalmanFilterEngine],
    authors="F. Capolupo",
    repo="https://github.com/FraCpl/KalmanFilterEngine.jl/blob/{commit}{path}#{line}",
    sitename="KalmanFilterEngine.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://FraCpl.github.io/KalmanFilterEngine.jl",
        edit_link="master",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/FraCpl/KalmanFilterEngine.jl",
    devbranch="master",
)
