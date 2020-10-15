module LRMoE

import Base: size, length, convert, show, getindex, rand, vec, inv
import Base: sum, maximum, minimum, ceil, floor, extrema, +, -, *, ==
import Base.Math: @horner

using StatsFuns
import StatsFuns: log1mexp, log1pexp, logsumexp

using Distributions
import Distributions: pdf, cdf, ccdf, logpdf, logcdf, logccdf
import Distributions: UnivariateDistribution, DiscreteUnivariateDistribution, ContinuousUnivariateDistribution
import Distributions: LogNormal


export
    # generic types
    ZeroInflation,
    ExpertSupport,
    AnyExpert,
    DiscreteExpert,
    ContinuousExpert,
    RealDiscreteExpert,
    RealContinuousExpert,
    NonNegDiscreteExpert,
    NonNegContinuousExpert,
    ZIDiscreteExpert,
    ZIContinuousExpert,
    NonZIDiscreteExpert,
    NonZIContinuousExpert,

    # loglikelihood functions
    pdf, logpdf,
    cdf, logcdf,

    expert_ll_pos,
    expert_tn_pos,
    expert_tn_bar_pos,
    expert_ll,
    expert_tn,
    expert_tn_bar,

    # gating
    LogitGating,

    # experts
    LogNormalExpert, ZILogNormalExpert,
    PoissonExpert, ZIPoissonExpert



### source files

include("utils.jl")

include("gating.jl")
include("expert.jl")

# include("experts/ll/expert_ll_pos.jl")

"""
A Julia package for the Logit-Reduced Mixture-of-Experts (LRMoE) model.
"""
LRMoE

end # module
