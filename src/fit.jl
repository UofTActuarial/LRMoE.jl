const fit = [
    "fit_main",
    "em",
    "fit_exact"
]

for dname in fit
    include(joinpath("fit", "$(dname).jl"))
end