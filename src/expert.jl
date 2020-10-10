# Abstract type: whether the expert is zero-inflated
abstract type ZeroInflation end
struct ZI <: ZeroInflation end
struct NonZI <: ZeroInflation end

abstract type ExpertSupport end
struct RealValued <: ExpertSupport end
struct NonNegative <: ExpertSupport end

# Abstract type: AnyExpert
abstract type AnyExpert{s<:ExpertSupport, z<:ZeroInflation, d<:UnivariateDistribution} end
# Discrete or Continuous
const DiscreteExpert{s<:ExpertSupport, z<:ZeroInflation} = AnyExpert{s, z, DiscreteUnivariateDistribution}
const ContinuousExpert{s<:ExpertSupport, z<:ZeroInflation} = AnyExpert{s, z, ContinuousUnivariateDistribution}
# Real-valued expert distributions
const RealDiscreteExpert = DiscreteExpert{RealValued, NonZI}
const RealContinuousExpert = ContinuousExpert{RealValued, NonZI}
# Nonnegative-valued expert distributions: actuarial-specific
const NonNegDiscreteExpert{z<:ZeroInflation} = DiscreteExpert{NonNegative, z}
const NonNegContinuousExpert{z<:ZeroInflation} = ContinuousExpert{NonNegative, z}
# Zero-inflated
const ZIDiscreteExpert = NonNegDiscreteExpert{ZI}
const ZIContinuousExpert = NonNegContinuousExpert{ZI}
# Non zero-inflated
const NonZIDiscreteExpert = NonNegDiscreteExpert{NonZI}
const NonZIContinuousExpert = NonNegContinuousExpert{NonZI}


##### specific distributions #####

const discrete_experts = [
    
]

const continuous_experts = [
    # NonZI
    "lognormal",
    # ZI
    "zilognormal"
]

for dname in discrete_experts
    include(joinpath("experts", "discrete", "$(dname).jl"))
end

for dname in continuous_experts
    include(joinpath("experts", "continuous", "$(dname).jl"))
end