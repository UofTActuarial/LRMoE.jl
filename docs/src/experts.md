# Expert Functions

The **LRMoE** package supports a collection of distributions commonly used for modelling insurance claim frequency and severitiy.

## Common Interface

Expert functions are implemented as subtypes of the `AnyExpert` type in this package.

Considering zero inflation is prominant in many actuarial applications, expert functions can be either zero-inflated or not.

```julia
# Abstract type: whether the expert is zero-inflated
abstract type ZeroInflation end
struct ZI <: ZeroInflation end
struct NonZI <: ZeroInflation end
```

Expert functions can be defined either on the real line, or only on nonnegative values (as is usually the case for actuarial loss modelling).

```julia
abstract type ExpertSupport end
struct RealValued <: ExpertSupport end
struct NonNegative <: ExpertSupport end
```

Each expert function is a univariate distribution (see `UnivariateDistribution` in `Distributions.jl`), with appropriate support and indication of
zero inflation. 
```julia
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
```

A good number of expert functions are simply wrappers around the `Distribution` type in `Distributions.jl` (details [here](https://juliastats.org/Distributions.jl/stable/types/)), and functions such as `pdf` and `cdf` are also directly using those in `Distributions.jl`.

Specific to actuarial application, a collection of commonly used distributions are also included in this package, e.g. Burr, GammaCount, etc.



## Continuous Distributions

```@docs
GammaExpert
```