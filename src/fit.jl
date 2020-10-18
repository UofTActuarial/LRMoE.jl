const fit = [
    "fit_main",
    "em"
]

for dname in fit
    include(joinpath("fit", "$(dname).jl"))
end