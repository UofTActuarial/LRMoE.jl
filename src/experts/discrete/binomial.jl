"""
    BinomialExpert(n, p)

PMF:

```math
P(X = k) = {n \\choose k}p^k(1-p)^{n-k},  \\quad \\text{ for } k = 0,1,2, \\ldots, n.
```

See also: [Binomial Distribution](https://en.wikipedia.org/wiki/Binomial_distribution) (Wikipedia) 

"""
struct BinomialExpert{T<:Real} <: NonZIDiscreteExpert
    n::Int
    p::T
    BinomialExpert{T}(n, p) where {T<:Real} = new{T}(n, p)
end

function BinomialExpert(n::Integer, p::T; check_args=true) where {T<:Real}
    check_args && @check_args(BinomialExpert, 0 <= p <= 1 && isa(n, Integer))
    return BinomialExpert{T}(n, p)
end

## Outer constructors
# BinomialExpert(n::Integer, p::Real) = BinomialExpert(n, float(p))
BinomialExpert(n::Integer, p::Integer) = BinomialExpert(n, float(p))
BinomialExpert() = BinomialExpert(2, 0.50)

## Conversion
function convert(::Type{BinomialExpert{T}}, n::Int, p::S) where {T<:Real,S<:Real}
    return BinomialExpert(n, T(p))
end
function convert(::Type{BinomialExpert{T}}, d::BinomialExpert{S}) where {T<:Real,S<:Real}
    return BinomialExpert(d.n, T(d.p); check_args=false)
end
copy(d::BinomialExpert) = BinomialExpert(d.n, d.p; check_args=false)

## Loglikelihood of Expoert
function logpdf(d::BinomialExpert, x...)
    return isinf(x...) ? 0.0 : Distributions.logpdf.(Distributions.Binomial(d.n, d.p), x...)
end
function pdf(d::BinomialExpert, x...)
    return isinf(x...) ? -Inf : Distributions.pdf.(Distributions.Binomial(d.n, d.p), x...)
end
function logcdf(d::BinomialExpert, x...)
    return isinf(x...) ? 0.0 : Distributions.logcdf.(Distributions.Binomial(d.n, d.p), x...)
end
function cdf(d::BinomialExpert, x...)
    return isinf(x...) ? 1.0 : Distributions.cdf.(Distributions.Binomial(d.n, d.p), x...)
end

## expert_ll, etc
expert_ll_exact(d::BinomialExpert, x::Real) = LRMoE.logpdf(d, x)
function expert_ll(d::BinomialExpert, tl::Real, yl::Real, yu::Real, tu::Real)
    expert_ll = if (yl == yu)
        logpdf.(d, yl)
    else
        logcdf.(d, yu) + log1mexp.(logcdf.(d, ceil.(yl) .- 1) - logcdf.(d, yu))
    end
    return expert_ll
end
function expert_tn(d::BinomialExpert, tl::Real, yl::Real, yu::Real, tu::Real)
    expert_tn = if (tl == tu)
        logpdf.(d, tl)
    else
        logcdf.(d, tu) + log1mexp.(logcdf.(d, ceil.(tl) .- 1) - logcdf.(d, tu))
    end
    return expert_tn
end
function expert_tn_bar(d::BinomialExpert, tl::Real, yl::Real, yu::Real, tu::Real)
    expert_tn_bar = if (tl == tu)
        log1mexp.(logpdf.(d, tl))
    else
        log1mexp.(logcdf.(d, tu) + log1mexp.(logcdf.(d, ceil.(tl) .- 1) - logcdf.(d, tu)))
    end
    return expert_tn_bar
end

exposurize_expert(d::BinomialExpert; exposure=1) = d

## Parameters
params(d::BinomialExpert) = (d.n, d.p)
function params_init(y, d::BinomialExpert)
    n_init = Int(maximum(vec(y))) + 2
    p_init = mean(y) / n_init
    try
        return BinomialExpert(n_init, p_init)
    catch
        return BinomialExpert()
    end
end

## Simululation
sim_expert(d::BinomialExpert) = Distributions.rand(Distributions.Binomial(d.n, d.p), 1)[1]

## penalty
penalty_init(d::BinomialExpert) = []
no_penalty_init(d::BinomialExpert) = []
penalize(d::BinomialExpert, p) = 0.0

## statistics
mean(d::BinomialExpert) = mean(Distributions.Binomial(d.n, d.p))
var(d::BinomialExpert) = var(Distributions.Binomial(d.n, d.p))
quantile(d::BinomialExpert, p) = quantile(Distributions.Binomial(d.n, d.p), p)

## Misc functions for E-Step

function _sum_densy_series(d::BinomialExpert, yl, yu)
    if isinf(yu)
        series = 0:(max(yl - 1, 0))
        return d.n * d.p - sum(pdf.(d, series) .* series)[1]
    else
        series = yl:(min(yu, d.n))
        return sum(pdf.(d, series) .* series)[1]
    end
end

function _int_obs_Y_raw(d::BinomialExpert, yl, yu)
    return _sum_densy_series(d, yl, yu)
end

function _int_lat_Y_raw(d::BinomialExpert, tl, tu)
    return max(d.n * d.p - _sum_densy_series(d, tl, tu), 0)
end

## EM: M-Step
function EM_M_expert(d::BinomialExpert,
    tl, yl, yu, tu,
    exposure,
    z_e_obs, z_e_lat, k_e;
    penalty=true, pen_pararms_jk=[2.0 1.0])

    # Not affected by Exposurize
    expert_ll_pos = expert_ll.(d, tl, yl, yu, tu)
    expert_tn_bar_pos = expert_tn_bar.(d, tl, yl, yu, tu)

    # Further E-Step
    yl_yu_unique = unique_bounds(yl, yu)
    int_obs_Y_tmp = _int_obs_Y_raw.(d, yl_yu_unique[:, 1], yl_yu_unique[:, 2])
    Y_e_obs =
        exp.(-expert_ll_pos) .*
        int_obs_Y_tmp[match_unique_bounds(hcat(vec(yl), vec(yu)), yl_yu_unique)]
    nan2num(Y_e_obs, 0.0) # get rid of NaN
    inf2num(Y_e_obs, 0.0) # get rid of inf

    tl_tu_unique = unique_bounds(tl, tu)
    int_lat_Y_tmp = _int_lat_Y_raw.(d, tl_tu_unique[:, 1], tl_tu_unique[:, 2])
    Y_e_lat =
        exp.(-expert_tn_bar_pos) .*
        int_lat_Y_tmp[match_unique_bounds(hcat(vec(tl), vec(tu)), tl_tu_unique)]
    nan2num(Y_e_lat, 0.0) # get rid of NaN
    inf2num(Y_e_lat, 0.0) # get rid of inf

    # Update parameters
    term_zkz_Y = (z_e_obs .* Y_e_obs) .+ (z_e_lat .* k_e .* Y_e_lat)
    term_zkz_n_Y = (z_e_obs .* (d.n .- Y_e_obs)) .+ (z_e_lat .* k_e .* (d.n .- Y_e_lat))

    p_new = sum(term_zkz_Y)[1] / (sum(term_zkz_Y)[1] + sum(term_zkz_n_Y)[1])

    # Deal with zero mass
    if ((1 - p_new)^d.n > 0.999999) || (isnan((1 - p_new)^d.n))
        p_new = d.p
    end

    return BinomialExpert(d.n, p_new)
end

## EM: M-Step, exact observations
function EM_M_expert_exact(d::BinomialExpert,
    ye, exposure,
    z_e_obs;
    penalty=true, pen_pararms_jk=[Inf 1.0 Inf])

    # Update parameters
    term_zkz_Y = z_e_obs .* ye
    term_zkz_n_Y = z_e_obs .* (d.n .- ye)

    p_new = sum(term_zkz_Y)[1] / (sum(term_zkz_Y)[1] + sum(term_zkz_n_Y)[1])

    # Deal with zero mass
    if (1 - p_new)^d.n > 0.999999 || (isnan((1 - p_new)^d.n))
        p_new = d.p
    end

    return BinomialExpert(d.n, p_new)
end