module LRMoE

import Base: size, length, convert, show, getindex, rand, vec, inv, expm1, abs
import Base: sum, maximum, minimum, ceil, floor, extrema, +, -, *, ==
import Base: convert, copy, findfirst
import Base.Math: @horner
import Base: π

using StatsFuns
import StatsFuns: log1mexp, log1pexp, logsumexp
import StatsFuns: sqrt2, invsqrt2π

using Statistics
import Statistics: quantile

using Distributions
import Distributions: pdf, cdf, ccdf, logpdf, logcdf, logccdf, quantile
import Distributions: rand, AbstractRNG
import Distributions: mean, var, skewness, kurtosis
import Distributions: UnivariateDistribution, DiscreteUnivariateDistribution, ContinuousUnivariateDistribution
import Distributions: @distr_support, RecursiveProbabilityEvaluator
import Distributions: Bernoulli, Multinomial
import Distributions: Binomial, Poisson
import Distributions: Gamma, InverseGaussian, LogNormal, Normal, Weibull

using InvertedIndices
import InvertedIndices: Not

using LinearAlgebra
import LinearAlgebra: I, Cholesky

using SpecialFunctions
import SpecialFunctions: erf, loggamma, gamma_inc, gamma

using QuadGK
import QuadGK: quadgk

using Optim
import Optim: optimize, minimizer

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

    loglik_aggre_dim,
    loglik_aggre_gate_dim,
    loglik_aggre_gate_dim_comp,
    loglik_np,
    loglik_exact,

    penalty_α,
    penalty_params,
    penalize,

    # EM related
    nan2num,
    EM_E_z_obs,
    EM_E_z_lat,
    EM_E_k,
    EM_M_α,
    EM_M_dQdα,
    EM_M_dQ2dα2,
    EM_E_z_zero_obs_update,
    EM_E_z_zero_lat_update,
    EM_E_z_zero_obs,
    EM_E_z_zero_lat,

    fit_main,


    # gating
    LogitGating,

    # experts
    GammaCount,

    params,
    GammaExpert, ZIGammaExpert,
    InverseGaussianExpert, ZIInverseGaussianExpert,
    LogNormalExpert, ZILogNormalExpert,
    WeibullExpert, ZIWeibullExpert,

    BinomialExpert, ZIBinomialExpert,
    GammaCountExpert, ZIGammaCountExpert,
    NegativeBinomialExpert, ZINegativeBinomialExpert,
    PoissonExpert, ZIPoissonExpert,

    # fitting
    fit_main,
    fit_exact,

    # simulation
    sim_expert,
    sim_logit_gating,
    sim_dataset


### source files

include("utils.jl")

include("gating.jl")
include("expert.jl")
include("loglik.jl")
include("penalty.jl")

include("fit.jl")

include("simulation.jl")

# include("experts/ll/expert_ll_pos.jl")

"""
A Julia package for the Logit-Reduced Mixture-of-Experts (LRMoE) model.
"""
LRMoE

end # module
