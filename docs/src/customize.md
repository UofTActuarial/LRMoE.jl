# [Adding Customized Expert Functions](@id customize_experts)

In addition to the expert functions included in the package, users can also write their own
expert functions specific to a particular modelling problem (not confined within actuarial contexts).

## Type Hierarchy

Expert functions are implemented as subtypes of the `AnyExpert` type in this package.
A good number of expert functions are simply wrappers around the `UnivariateDistribution` type in `Distributions.jl` (details [here](https://juliastats.org/Distributions.jl/stable/types/)), and functions such as `pdf` and `cdf` are also directly using those in `Distributions.jl`.

Specific to actuarial application, a collection of commonly used distributions are also included in this package, e.g. Burr, GammaCount, etc. See below for details.

Expert functions can be defined either on the real line, or only on nonnegative values (as is usually the case for actuarial loss modelling).

```julia
# Abstract type: support of expert function
abstract type ExpertSupport end
struct RealValued <: ExpertSupport end
struct NonNegative <: ExpertSupport end
```

Considering zero inflation is prominant in many actuarial applications, expert functions can be either zero-inflated or not,
provided that they are supported only on nonnegative values.

```julia
# Abstract type: whether the expert is zero-inflated
abstract type ZeroInflation end
struct ZI <: ZeroInflation end
struct NonZI <: ZeroInflation end
```

Each expert function is a univariate distribution (see [here](https://juliastats.org/Distributions.jl/stable/univariate/) for `UnivariateDistribution` in `Distributions.jl`), with appropriate indication of support and zero inflation. 

```julia
# Abstract type: AnyExpert
abstract type AnyExpert{s<:ExpertSupport, z<:ZeroInflation, d<:UnivariateDistribution} end
# Discrete or Continuous
const DiscreteExpert{s<:ExpertSupport, z<:ZeroInflation} = 
                AnyExpert{s, z, DiscreteUnivariateDistribution}
const ContinuousExpert{s<:ExpertSupport, z<:ZeroInflation} = 
                AnyExpert{s, z, ContinuousUnivariateDistribution}
# Real-valued expert distributions
const RealDiscreteExpert = DiscreteExpert{RealValued, NonZI}
const RealContinuousExpert = ContinuousExpert{RealValued, NonZI}
# Nonnegative-valued expert distributions
const NonNegDiscreteExpert{z<:ZeroInflation} = DiscreteExpert{NonNegative, z}
const NonNegContinuousExpert{z<:ZeroInflation} = ContinuousExpert{NonNegative, z}
# Zero-inflated
const ZIDiscreteExpert = NonNegDiscreteExpert{ZI}
const ZIContinuousExpert = NonNegContinuousExpert{ZI}
# Non zero-inflated
const NonZIDiscreteExpert = NonNegDiscreteExpert{NonZI}
const NonZIContinuousExpert = NonNegContinuousExpert{NonZI}
```
