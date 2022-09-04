# Abstract type: whether the expert is zero-inflated
abstract type ZeroInflation end
struct ZI <: ZeroInflation end
struct NonZI <: ZeroInflation end

abstract type ExpertSupport end
struct RealValued <: ExpertSupport end
struct NonNegative <: ExpertSupport end

# Abstract type: AnyExpert
abstract type AnyExpert{s<:ExpertSupport,z<:ZeroInflation,d<:UnivariateDistribution} end
# Discrete or Continuous
const DiscreteExpert{s<:ExpertSupport,z<:ZeroInflation} = AnyExpert{
    s,z,DiscreteUnivariateDistribution
}
const ContinuousExpert{s<:ExpertSupport,z<:ZeroInflation} = AnyExpert{
    s,z,ContinuousUnivariateDistribution
}
# Real-valued expert distributions
const RealDiscreteExpert = DiscreteExpert{RealValued,NonZI}
const RealContinuousExpert = ContinuousExpert{RealValued,NonZI}
# Nonnegative-valued expert distributions: actuarial-specific
const NonNegDiscreteExpert{z<:ZeroInflation} = DiscreteExpert{NonNegative,z}
const NonNegContinuousExpert{z<:ZeroInflation} = ContinuousExpert{NonNegative,z}
# Zero-inflated
const ZIDiscreteExpert = NonNegDiscreteExpert{ZI}
const ZIContinuousExpert = NonNegContinuousExpert{ZI}
# Non zero-inflated
const NonZIDiscreteExpert = NonNegDiscreteExpert{NonZI}
const NonZIContinuousExpert = NonNegContinuousExpert{NonZI}

## Broadcast issues
Broadcast.broadcastable(d::RealDiscreteExpert) = Ref(d)
Broadcast.broadcastable(d::RealContinuousExpert) = Ref(d)
Broadcast.broadcastable(d::ZIDiscreteExpert) = Ref(d)
Broadcast.broadcastable(d::ZIContinuousExpert) = Ref(d)
Broadcast.broadcastable(d::NonZIDiscreteExpert) = Ref(d)
Broadcast.broadcastable(d::NonZIContinuousExpert) = Ref(d)

##### specific distributions #####

const add_experts = [
    "burr",
    "gammacount",
]

const discrete_experts = [
    "binomial", "zibinomial",
    "gammacount", "zigammacount",
    "negativebinomial", "zinegativebinomial",
    "poisson", "zipoisson",
]

const continuous_experts = [
    "burr", "ziburr",
    "gamma", "zigamma",
    "inversegaussian", "ziinversegaussian",
    "lognormal", "zilognormal",
    "weibull", "ziweibull",]

const calcs = [
    # "expert_ll_pos",
    # "expert_tn_pos",
    # "expert_tn_bar_pos",
    # "expert_ll",
    # "expert_tn",
    # "expert_tn_bar",
    "exposurize"
]

for dname in add_experts
    include(joinpath("experts", "add_dist", "$(dname).jl"))
end

for dname in discrete_experts
    include(joinpath("experts", "discrete", "$(dname).jl"))
end

for dname in continuous_experts
    include(joinpath("experts", "continuous", "$(dname).jl"))
end

for dname in calcs
    include(joinpath("experts", "ll", "$(dname).jl"))
end