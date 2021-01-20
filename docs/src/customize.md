# [Adding Customized Expert Functions](@id customize_experts)

In addition to the expert functions included in the package, users can also write their own
expert functions specific to a particular modelling problem (not confined within actuarial contexts).

## Type Hierarchy

Expert functions are implemented as subtypes of the `AnyExpert` type in this package.
A good number of expert functions are simply wrappers around the `UnivariateDistribution` type in `Distributions.jl` (details [here](https://juliastats.org/Distributions.jl/stable/types/)), and functions such as `pdf` and `cdf` are also directly using those in `Distributions.jl`.

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

## Example: Gamma Expert

As an example, we will use the Gamma expert to demonstrate how to write a customized expert function
for any continuous distribution. For discrete distributions, the Poisson distribution is a good example,
but we will omit the details here. In this illustrative example, we will go through the source code of
Gamma Expert (available [here](https://github.com/sparktseung/LRMoE.jl/blob/main/src/experts/continuous/gamma.jl))
to see what is needed to add a customized expert function.

Before starting, we create a new source code file `gamma.jl` for Gamma expert, and 
put it in the corresponding folder, i.e. `src/experts/continuous`.
### Defining a struct

The first step is to define the Gamma expert function and some basic and common functions shared by all experts.
This corresponds to the first block of the source code. The details are quite similar to 
adding a customized distribution to `Distributions.jl`, including defining the new `GammaExpert` as a sub-type of `NonZIContinuousExpert`, 
some constructors and conversions of parameters.

```julia
struct GammaExpert{T<:Real} <: NonZIContinuousExpert
    k::T
    θ::T
    GammaExpert{T}(k::T, θ::T) where {T<:Real} = new{T}(k, θ)
end

function GammaExpert(k::T, θ::T; check_args=true) where {T <: Real}
    check_args && @check_args(GammaExpert, k >= zero(k) && θ > zero(θ))
    return GammaExpert{T}(k, θ)
end

## Outer constructors
GammaExpert(k::Real, θ::Real) = GammaExpert(promote(k, θ)...)
GammaExpert(k::Integer, θ::Integer) = GammaExpert(float(k), float(θ))
GammaExpert() = GammaExpert(1.0, 1.0)

## Conversion
function convert(::Type{GammaExpert{T}}, k::S, θ::S) where {T <: Real, S <: Real}
    GammaExpert(T(k), T(θ))
end
function convert(::Type{GammaExpert{T}}, d::GammaExpert{S}) where {T <: Real, S <: Real}
    GammaExpert(T(d.k), T(d.θ), check_args=false)
end
copy(d::GammaExpert) = GammaExpert(d.k, d.θ, check_args=false)
```
### Exporting the expert function


This step should technically be at the very end, but we introduce it here to facilitate testing during development.
In order for the expert function to be accessible to the user, it should be exported by modifying the files
`src/experts/expert.jl` and `src/LRMoE.jl`: In the former, `gamma` is added to `contintous_experts` so that the
source file `src/experts/continuous/gamma.jl` is included; In the latter, `GammaExpert` is exported so that
the gamma expert function is accessible outside of the package.

### Basic and additional functions

For all expert functions, there are a number of basic functions absolutely needed for using the package (highlighted in **bold**).
In addition, some functions may be omitted (e.g. calculating limited expected value) if they are not
relevant to the modeling problem at hand.

* **Basic probability functions**: `pdf`, `logpdf`, `cdf` and `logcdf`. These are used for calculating the loglikelihood
of the model. Notice that for `logpdf` etc., we have directly used the corresponding functions in `Distributions.jl`
since the Gamma distribution is already implemented there. If this is not the case, the user can also
add a new distribution type following the guide in `Distributions.jl`, and add the source code to the folder `src/experts/add_dist`.

```julia
## Loglikelihood of Expert
logpdf(d::GammaExpert, x...) = (d.k < 1 && x... <= 0.0) ? -Inf : Distributions.logpdf.(Distributions.Gamma(d.k, d.θ), x...)
pdf(d::GammaExpert, x...) = (d.k < 1 && x... <= 0.0) ? 0.0 : Distributions.pdf.(Distributions.Gamma(d.k, d.θ), x...)
logcdf(d::GammaExpert, x...) = (d.k < 1 && x... <= 0.0) ? -Inf : Distributions.logcdf.(Distributions.Gamma(d.k, d.θ), x...)
cdf(d::GammaExpert, x...) = (d.k < 1 && x... <= 0.0) ? 0.0 : Distributions.cdf.(Distributions.Gamma(d.k, d.θ), x...)
```



### Maximizing loglikelihood: exact observation

The next step is to derive the loglikelihood of a Gamma distribution, which is given in Section 3.2 of [Fung et al. (2019)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3740061). Notice that the cited paper considers the issue of data truncation and censoring,
but as a first step, we will just assume all data are observed exactly. Data truncation and censoring are dealt with
afterwards.

