module LRMoE

import Base: size, length, convert, show, getindex, rand, vec, inv, expm1, abs, log1p
import Base: isnan, isinf
import Base: sum, maximum, minimum, ceil, floor, extrema, +, -, *, ==
import Base: convert, copy, findfirst, summary
import Base.Math: @horner
import Base: π
import Base.Threads: @threads, nthreads, threadid

using StatsFuns
import StatsFuns: log1mexp, log1pexp, logsumexp
import StatsFuns: sqrt2, invsqrt2π

using Statistics
import Statistics: quantile, mean, var, median

using Distributions
import Distributions: pdf, cdf, ccdf, logpdf, logcdf, logccdf, quantile
import Distributions: rand, AbstractRNG
import Distributions: mean, var, skewness, kurtosis
import Distributions:
    UnivariateDistribution, DiscreteUnivariateDistribution, ContinuousUnivariateDistribution
import Distributions: @distr_support, RecursiveProbabilityEvaluator
import Distributions: Bernoulli, Multinomial
import Distributions: Binomial, Poisson
import Distributions: Gamma, InverseGaussian, LogNormal, Normal, Weibull

using InvertedIndices
import InvertedIndices: Not

using LinearAlgebra
import LinearAlgebra: I, Cholesky

using SpecialFunctions
import SpecialFunctions: erf, loggamma, gamma_inc, gamma, beta_inc

using QuadGK
import QuadGK: quadgk

using Optim
import Optim: optimize, minimizer

using Clustering
import Clustering: kmeans, assignments, counts

using HypothesisTests
import HypothesisTests: ExactOneSampleKSTest, pvalue, ksstats

using Logging

using Roots
import Roots: find_zero, Order2

export
    ## generic types
    # convert,
    # copy,
    # ZeroInflation,
    # ExpertSupport,
    # AnyExpert,
    # DiscreteExpert,
    # ContinuousExpert,
    # RealDiscreteExpert,
    # RealContinuousExpert,
    # NonNegDiscreteExpert,
    # NonNegContinuousExpert,
    # ZIDiscreteExpert,
    # ZIContinuousExpert,
    # NonZIDiscreteExpert,
    # NonZIContinuousExpert,

    ## model related
    summary,
    # LRMoEModel,
    # LRMoESTD,
    # LRMoEFittingResult,
    # LRMoESTDFit,

    ## loglikelihood functions
    pdf, logpdf,
    cdf, logcdf,
    rowlogsumexp,
    expert_ll,
    expert_tn,
    expert_tn_bar,
    # expert_ll_list,
    # expert_tn_list,
    # expert_tn_bar_list, loglik_aggre_dim,
    # loglik_aggre_gate_dim,
    # loglik_aggre_gate_dim_comp,
    # loglik_np,
    # loglik_exact, penalty_α,
    # penalty_params,
    # penalize,

    ## EM related
    # nan2num,
    # EM_E_z_obs,
    # EM_E_z_lat,
    # EM_E_k,
    # EM_M_α,
    # EM_M_dQdα,
    # EM_M_dQ2dα2,
    # EM_E_z_zero_obs_update,
    # EM_E_z_zero_lat_update,
    # EM_E_z_zero_obs,
    # EM_E_z_zero_lat,

    ## init
    cmm_init,
    cmm_init_exact,

    ## gating
    LogitGating,

    ## experts
    Burr,
    GammaCount, params,
    p_zero,
    params_init,
    exposurize_expert,
    exposurize_model, BurrExpert, ZIBurrExpert,
    GammaExpert, ZIGammaExpert,
    InverseGaussianExpert, ZIInverseGaussianExpert,
    LogNormalExpert, ZILogNormalExpert,
    WeibullExpert, ZIWeibullExpert, BinomialExpert, ZIBinomialExpert,
    GammaCountExpert, ZIGammaCountExpert,
    NegativeBinomialExpert, ZINegativeBinomialExpert,
    PoissonExpert, ZIPoissonExpert,

    ## fitting
    fit_main,
    fit_exact,
    fit_LRMoE,

    ## simulation
    sim_expert,
    sim_logit_gating,
    sim_dataset,

    # prediction
    mean, var, quantile,
    predict_class_prior,
    predict_class_posterior,
    predict_mean_prior,
    predict_mean_posterior,
    predict_var_prior,
    predict_var_posterior,
    predict_limit_prior,
    predict_limit_posterior,
    predict_excess_prior,
    predict_excess_posterior,
    predict_VaRCTE_prior,
    predict_VaRCTE_posterior

### source files

include("utils.jl")
include("AICBIC.jl")
include("modelstruct.jl")

include("gating.jl")
include("expert.jl")
include("loglik.jl")
include("penalty.jl")

include("paramsinit.jl")
include("fit.jl")

include("simulation.jl")
include("predict.jl")

# include("experts/ll/expert_ll_pos.jl")

"""
A Julia package for the Logit-Reduced Mixture-of-Experts (LRMoE) model.
"""
LRMoE

end # module