module LRMoE

import Base: size, length, convert, show, getindex, rand, vec, inv
import Base: sum, maximum, minimum, extrema, +, -, *, ==
import Base.Math: @horner

using StatsFuns
import StatsFuns: log1mexp, log1pexp

using Distributions
import Distributions: pdf, cdf, ccdf, logpdf, logcdf, logccdf
import Distributions: UnivariateDistribution, DiscreteUnivariateDistribution, ContinuousUnivariateDistribution
import Distributions: LogNormal


"""
A Julia package for the Logit-Reduced Mixture-of-Experts (LRMoE) model.
"""
LRMoE

end # module
