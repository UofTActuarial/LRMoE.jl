const fit = [
    "fit_main",
    "em",
    "fit_exact",
    "fit_interface",
]

for dname in fit
    include(joinpath("fit", "$(dname).jl"))
end