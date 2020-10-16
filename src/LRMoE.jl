module LRMoE

import Base: size, length, convert, show, getindex, rand, vec, inv
import Base: sum, maximum, minimum, ceil, floor, extrema, +, -, *, ==
import Base: convert, copy
import Base.Math: @horner

using StatsFuns
import StatsFuns: log1mexp, log1pexp, logsumexp

using Distributions
import Distributions: pdf, cdf, ccdf, logpdf, logcdf, logccdf
import Distributions: UnivariateDistribution, DiscreteUnivariateDistribution, ContinuousUnivariateDistribution
import Distributions: LogNormal


export
    # generic types
    convert,
    copy,
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
    rowlogsumexp,

    expert_ll_pos,
    expert_tn_pos,
    expert_tn_bar_pos,
    expert_ll,
    expert_tn,
    expert_tn_bar,
    expert_ll_pos_list,
    expert_tn_pos_list,
    expert_tn_bar_pos_list,
    expert_ll_list,
    expert_tn_list,
    expert_tn_bar_list,

    # gating
    LogitGating,

    # experts
    LogNormalExpert, ZILogNormalExpert,
    PoissonExpert, ZIPoissonExpert



### source files

include("utils.jl")

include("gating.jl")
include("expert.jl")
include("loglik.jl")

# include("experts/ll/expert_ll_pos.jl")

"""
A Julia package for the Logit-Reduced Mixture-of-Experts (LRMoE) model.
"""
LRMoE

end # module
