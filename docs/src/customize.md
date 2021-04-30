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

For all expert functions, there are a number of basic functions absolutely needed for using the package.
In addition, some functions may be omitted (e.g. calculating limited expected value) if they are not
relevant to the modeling problem at hand.

**Basic probability functions**: `pdf`, `logpdf`, `cdf` and `logcdf`. These are used for calculating the loglikelihood
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

**Parameters and initialization**: The following functions are necessary for calling the functions to initialize parameters (e.g. `cmm_init_params`).

In the following `params_init` function, we assume a vector of observations `y`,
and match the first two moments to solve for the parameters of a gamma distribution. Notice that the function should return an expert function, not the
parameter values. Also, an empty expert should also be added to the corresponding list in `src/paramsinit.jl`. In this case, `GammaExpert()` is added to `_default_expert_continuous`.

The function `ks_distance` is used to select an initialization of expert which has
the lowest test statistics for the Kolmogorov-Smirnov test, in other words, the
K-S distance is minimized. The `ks_distance` simply calculates the test statistics
given a vector of observations `y` and an expert function `GammaExpert`
(i.e. to see if the observations `y` come from a Gamma distribution).

```julia
## Parameters
params(d::GammaExpert) = (d.k, d.θ)
function params_init(y, d::GammaExpert)
    pos_idx = (y .> 0.0)
    μ, σ2 = mean(y[pos_idx]), var(y[pos_idx])
    θ_init = σ2/μ
    k_init = μ/θ_init
    if isnan(θ_init) || isnan(k_init)
        return GammaExpert()
    else
        return GammaExpert(k_init, θ_init)
    end
end

## KS stats for parameter initialization
function ks_distance(y, d::GammaExpert)
    p_zero = sum(y .== 0.0) / sum(y .>= 0.0)
    return max(abs(p_zero-0.0), (1-0.0)*HypothesisTests.ksstats(y[y .> 0.0], Distributions.Gamma(d.k, d.θ))[2])
end
```

**Simulation**: A simulator is also needed, which simulates a vector of length `sample_size` from the expert function.

```julia
## Simululation
sim_expert(d::GammaExpert, sample_size) = Distributions.rand(Distributions.Gamma(d.k, d.θ), sample_size)
```

**Penalty on parameters**: This is also required. When fitting mixture models, the EM algorithm
may converge to a spurious model with extremely large or small parameter values,
which is undesirable for a number of reasons (e.g. giving infinite likelihood, or
straight up a NaN error). Hence, a penalty is imposed for these extreme cases.
In implementation, we essentially assume the parameters have a prior distribution
described by some hyperparameters. Consequently, the EM step is essentially
maximizing a posterior loglikelihood.

As a rule of thumb, we assume a Gamma prior for positive parameters and a Normal prior
for real parameters. For example, the penalty terms for Gamma experts are coded as follows. 

```julia
## penalty
penalty_init(d::GammaExpert) = [2.0 10.0 2.0 10.0]
no_penalty_init(d::GammaExpert) = [1.0 Inf 1.0 Inf]
penalize(d::GammaExpert, p) = (p[1]-1)*log(d.k) - d.k/p[2] + (p[3]-1)*log(d.θ) - d.θ/p[4]
```

In the function `penalize`, the hyperparameters are given as a vector `p`.
We assume the shape parameter `k` of the Gamma expert follows a Gamma prior distribution with shape `p[1]` and scale `p[2]`. Analogously, the scale parameter
`θ` of the Gamma expert follows a Gamma prior distribution with shape `p[3]` and scale `p[4]`. The `penalize` function calculates the prior logpdf of the
parameters, excluding the constant terms since they are irrelevant to the EM algorithm.

The functions `penalty_init` initializes some default hyperparameters, if they
are not given by the user. Similarly, `no_penalty_init` speficies hyperparameters
which poses no penalty, as plugging ion `p = [1.0 Inf 1.0 Inf]` into the
`penalize` function yields zero. Still, it is recommended to always use some penalty
in application to avoid the issue of spurious models mentioned above.

Note that the penalty term should be subtracted from the loglikelihood
when implementing the M-step of the EM algorithm (see below).

**Miscellaneous functions**: There are a number of miscellaneous functions implemented for the expert function as shown below. They are mainly used in 
predictive functions such as `predict_mean_prior`. It is recommended to code them
as well when adding a new expert function, but (some or all of) these functions can be optional if
the user only wants to fit an LRMoE model.

```julia
## statistics
mean(d::GammaExpert) = mean(Distributions.Gamma(d.k, d.θ))
var(d::GammaExpert) = var(Distributions.Gamma(d.k, d.θ))
quantile(d::GammaExpert, p) = quantile(Distributions.Gamma(d.k, d.θ), p)
lev(d::GammaExpert, u) = d.θ*d.k*gamma_inc(float(d.k+1), u/d.θ, 0)[1] + u*(1-gamma_inc(float(d.k), u/d.θ,0)[1])
excess(d::GammaExpert, u) = mean(d) - lev(d, u)
```





### M-Step: exact observation

The next step is to implement the M-step of the EM algorithm, which is given in Section 3.2 of [Fung et al. (2019)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3740061). Notice that the cited paper considers the issue of data truncation and censoring,
but as a first step, we will just assume all data are observed exactly. Data truncation and censoring are dealt with
afterwards.

The M-step with exact observation is carried out by the function
`EM_M_expert_exact` in the source code. Note that the function name and arguments
should not be altered, since it is referenced in the main fitting function of the
package.

In short, the goal is to maximize the objective function  `z_e_obs` * ``LL``, where ``LL`` is
the loglikelihood of the expert `d` when observing a vector `ye`. If the function argument
`penalty` is true, then a penalty term given by hyperparameters `pen_params_jk` (see also above) is subtracted from the objective function. The function argument
`expert_ll_pos` is a legacy and irrelevant to the M-step when observations `ye` are exact.

For Gamma expert, we are maximizing with respect to ``k`` and ``\lambda`` (without penalty) the objective function `z_e_obs` multiplied by

``LL = \sum_{i = 1}^{n} (k-1)\log(y_i) - y_i/k - \log(\Gamma(k)) - k\log(\lambda)``.

The optimization is done by calling `minimizer` in `Optim.jl`, which is ommited in this document. Penalty can also be added in the objective function if
the function argument `penalty` is set to `true`.

Finally, the `EM_M_expert_exact` returns a Gamma expert object containing
the updated parameters, thus completing the M-step.

### M-Step: censored and truncated observation

One important feature of this package is to deal with censored and truncated data,
which is common both in and outside actuarial contexts. In this case, the
loglikelihood is incomplete, in the sense that ``y_i`` above is not known exactly,
leading to an additional E-step before the M-step.

In particular, the M-step with data censoring and truncation is done by the 
function `EM_M_step`, which takes a few more arguments compared with 
`EM_M_expert_exact`. Again, the function name and arguments should not be altered.

Recall that we assume the obsercations are truncated between `tl` and `tu`, as well as censored between `yl` and `yu`. These lower and upper bounds are needed for the
M-step.

Censored data points are in the observed dataset. For all the censored observation `y`, we don't know their exact values, but only know that they are between 
`yl` and `yu`. Hence, for those observations, we should take the expected value
of the loglikelihood, further normalizing them by the probability of falling within
this interval (which is equal to `exp.(-expert_ll_pos)`).

For the Gamma expert, we refer to the equation of ``LL`` above: the uncertain terms
are ``\log(y_i)`` and ``y_i``, so we should compute their conditional expectation,
given that the exact observation is between `yl` and `yu`. This corresponds to
some numerical integration procedures in the source code. Note that the conditional
expectation should be computed with respect to the probability measure
implied by the old (i.e. not yet updated) parameters.

A similar remark can be made about truncated observations, which are not present
in the dataset due to truncation. Those are either smaller than the lower bound
of truncation `tl`, or larger than the upper bound of truncation `tu`. In addition,
the function argument `k_e` is equal to the expected number of lost observations
due to truncation, which should also be multiplied to the conditional expectation
of loglikelihood.

To be more specific, we also numerically integrate ``\log(y_i)`` and ``y_i`` in ``LL``, but from `0` to `tl` and from `tu` to `Inf`. Then, we normalize it by the probability `exp.(-expert_tn_bar_pos)` to obtain the conditional expectation.
Finally, the loglikelihood is further multiplied by `k_e` to account for all
unobserved data points due to truncation.

Finally, the loglikelihood of truncated observations is added to the objective function for optimization.

## Notes and tips

There are quite a few things to note when adding a customized expert function.

* Exact observations: If the problem at hand only concerns exact observations,
  then the user can write the `EM_M_exact` function only.

* Speeding up numerical integration: In `EM_M_step`, numerical integration
  for certain expert functions can take a long time. In the `gamma.jl` example,
  the author has first looked up for unique lower and upper bounds, and only integrate using those pairs. Then, the numerical integration is mapped to the original dataset. Users are advised to also take this approach, which can be done by conveniently copying and modifying the source code of `gamma.jl`.

* Optimization constraits: Some expert functions pose constrants on the parameters,
  e.g. the parameters of Gamma experts should be positive. However, constrained optimization can be computationally slow. In `gamma.jl`, we have used a log transformation, so that the optimization procedure `Optim.optimize` is unconstrained. Also, specifying a small search interval speeds things up.

* Zero inflation: For zero-inflated expert functions, the M-step should only
  take in positive observations for continuous experts, but all non-negative observations for discrete experts. See `gamma.jl` and `poisson.jl` for a comparison.

For further questions and tips when customizing expert functions, please contact the author on github.



