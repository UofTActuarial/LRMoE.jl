# Expert Functions

The **LRMoE.jl** package supports a collection of distributions commonly used for modelling insurance claim frequency and severity.

## Discrete Distributions (Frequency Modelling)

```@docs
BinomialExpert
NegativeBinomialExpert
PoissonExpert
GammaCountExpert
```

## Continuous Distributions (Severity Modelling)

```@docs
BurrExpert
GammaExpert
InverseGaussianExpert
LogNormalExpert
WeibullExpert
```

## Zero Inflation

Zero inflation is supported for all discrete and continuous experts. They can be constructed by adding `ZI` in front of
an expert function, with an additional parameter `p` (or `p0` if the expert already uses `p`, e.g. binomial) for modelling a probability
mass at zero. Zero-inflated experts are used in the same way as their non-zero-inflated counterpart. A complete list of 
zero-inflated expert functions is given below.

```julia
ZIBinomialExpert(p0, n, p)
ZINegativeBinomialExpert(p0, n, p)
ZIPoissonExpert(p, λ)
ZIGammaCountExpert(p, m, s)
```

```julia
ZIBurrExpert(p, k, c, λ)
ZIGammaExpert(p, k, θ)
ZIInverseGaussianExpert(p, μ, λ)
ZILogNormalExpert(p, μ, σ)
ZIWeibullExpert(p, k, θ)
```

## Adding Customized Expert Functions

See [here](@ref customize_experts).