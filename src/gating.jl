# Abstract type: whether the gating function has random effects
# abstract type RandomGating end
# struct HasRE <: RandomGating end
# struct NonRE <: RandomGating end

# # Abstract type: AnyGating
# abstract type AnyGating{s<:RandomGating} end
# # Logit Gating
# const NonRandomGating = AnyGating{NonRE}

const gating = [
    "logit"
]

for dname in gating
    include(joinpath("gating", "$(dname).jl"))
end
