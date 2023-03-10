"""
    LogNormalExpert(μ, σ)

PDF:

```math
f(x; \\mu, \\sigma) = \\frac{1}{x \\sqrt{2 \\pi \\sigma^2}}
\\exp \\left( - \\frac{(\\log(x) - \\mu)^2}{2 \\sigma^2} \\right),
\\quad x > 0
```

See also: [Lognormal Distribution](https://en.wikipedia.org/wiki/Log-normal_distribution) (Wikipedia)

"""
struct LogNormalExpert{T<:Real} <: NonZIContinuousExpert
    μ::T
    σ::T
    LogNormalExpert{T}(µ::T, σ::T) where {T<:Real} = new{T}(µ, σ)
end

function LogNormalExpert(μ::T, σ::T; check_args=true) where {T<:Real}
    check_args && @check_args(LogNormalExpert, σ >= zero(σ))
    return LogNormalExpert{T}(μ, σ)
end

## Outer constructors
LogNormalExpert(μ::Real, σ::Real) = LogNormalExpert(promote(μ, σ)...)
LogNormalExpert(μ::Integer, σ::Integer) = LogNormalExpert(float(μ), float(σ))
LogNormalExpert() = LogNormalExpert(0.0, 1.0)

## Conversion
function convert(::Type{LogNormalExpert{T}}, μ::S, σ::S) where {T<:Real,S<:Real}
    return LogNormalExpert(T(μ), T(σ))
end
function convert(::Type{LogNormalExpert{T}}, d::LogNormalExpert{S}) where {T<:Real,S<:Real}
    return LogNormalExpert(T(d.μ), T(d.σ); check_args=false)
end
copy(d::LogNormalExpert) = LogNormalExpert(d.μ, d.σ; check_args=false)

## Loglikelihood of Expert
function logpdf(d::LogNormalExpert, x...)
    return Distributions.logpdf.(Distributions.LogNormal(d.μ, d.σ), x...)
end
pdf(d::LogNormalExpert, x...) = Distributions.pdf.(Distributions.LogNormal(d.μ, d.σ), x...)
function logcdf(d::LogNormalExpert, x...)
    return Distributions.logcdf.(Distributions.LogNormal(d.μ, d.σ), x...)
end
cdf(d::LogNormalExpert, x...) = Distributions.cdf.(Distributions.LogNormal(d.μ, d.σ), x...)

## expert_ll, etc
expert_ll_exact(d::LogNormalExpert, x::Real) = LRMoE.logpdf(d, x)
function expert_ll(d::LogNormalExpert, tl::Real, yl::Real, yu::Real, tu::Real)
    expert_ll = if (yl == yu)
        logpdf.(d, yl)
    else
        logcdf.(d, yu) + log1mexp.(logcdf.(d, yl) - logcdf.(d, yu))
    end
    expert_ll = (tu == 0.0) ? -Inf : expert_ll
    return expert_ll
end
function expert_tn(d::LogNormalExpert, tl::Real, yl::Real, yu::Real, tu::Real)
    expert_tn = if (tl == tu)
        logpdf.(d, tl)
    else
        logcdf.(d, tu) + log1mexp.(logcdf.(d, tl) - logcdf.(d, tu))
    end
    expert_tn = (tu == 0.0) ? -Inf : expert_tn
    return expert_tn
end
function expert_tn_bar(d::LogNormalExpert, tl::Real, yl::Real, yu::Real, tu::Real)
    expert_tn_bar = if (tl == tu)
        0.0
    else
        log1mexp.(logcdf.(d, tu) + log1mexp.(logcdf.(d, tl) - logcdf.(d, tu)))
    end
    return expert_tn_bar
end

exposurize_expert(d::LogNormalExpert; exposure=1) = d

## Parameters
params(d::LogNormalExpert) = (d.μ, d.σ)
function params_init(y, d::LogNormalExpert)
    pos_idx = (y .> 0.0)
    μ_init, σ_init = mean(log.(y[pos_idx])), sqrt(var(log.(y[pos_idx])))
    μ_init = isnan(μ_init) ? 0.0 : μ_init
    σ_init = isnan(σ_init) ? 1.0 : σ_init
    return LogNormalExpert(μ_init, σ_init)
end

## KS stats for parameter initialization
function ks_distance(y, d::LogNormalExpert)
    p_zero = sum(y .== 0.0) / sum(y .>= 0.0)
    return max(
        abs(p_zero - 0.0),
        (1 - 0.0) *
        HypothesisTests.ksstats(y[y .> 0.0], Distributions.LogNormal(d.μ, d.σ))[2],
    )
end

## Simululation
sim_expert(d::LogNormalExpert) = Distributions.rand(Distributions.LogNormal(d.μ, d.σ), 1)[1]

## penalty
penalty_init(d::LogNormalExpert) = [2.0 2.0]
no_penalty_init(d::LogNormalExpert) = [1.0 1.0]
penalize(d::LogNormalExpert, p) = -0.5 * (p[1] - 1) / (d.σ * d.σ) - (p[2] - 1) * log(d.σ)

## statistics
mean(d::LogNormalExpert) = mean(Distributions.LogNormal(d.μ, d.σ))
var(d::LogNormalExpert) = var(Distributions.LogNormal(d.μ, d.σ))
quantile(d::LogNormalExpert, p) = quantile(Distributions.LogNormal(d.μ, d.σ), p)
function lev(d::LogNormalExpert, u)
    if isinf(u)
        return mean(d)
    else
        return exp(d.μ + 0.5 * d.σ^2) * cdf.(Normal(d.μ + d.σ^2, d.σ), log(u)) +
               u * (1 - cdf.(Normal(d.μ, d.σ), log(u)))
    end
end
excess(d::LogNormalExpert, u) = mean(d) - lev(d, u)

## Misc functions for E-Step
function _diff_dens_series(d::LogNormalExpert, yl, yu)
    return exp(-0.5 * (log(yl) - d.μ)^2 / (d.σ^2)) - exp(-0.5 * (log(yu) - d.μ)^2 / (d.σ^2))
end

function _diff_dist_series(d::LogNormalExpert, yl, yu)
    return (0.5 + 0.5 * erf((log(yu) - d.μ) / (sqrt2 * d.σ))) -
           (0.5 + 0.5 * erf((log(yl) - d.μ) / (sqrt2 * d.σ)))
end

function _int_obs_logY(d::LogNormalExpert, yl, yu, expert_ll)
    if yl == yu
        return log(yl)
    else
        return exp(-expert_ll) * (
            d.σ * invsqrt2π * _diff_dens_series(d, yl, yu) +
            d.μ * _diff_dist_series(d, yl, yu)
        )
    end
end

function _int_lat_logY(d::LogNormalExpert, tl, tu, expert_tn_bar)
    return exp(-expert_tn_bar) * (
        d.μ - (
            d.σ * invsqrt2π * _diff_dens_series(d, tl, tu) +
            d.μ * _diff_dist_series(d, tl, tu)
        )
    )
end

function _zdensz_series(z)
    return (isinf(z) ? 0.0 : z * exp(-0.5 * z^2))
end

function _diff_zdensz_series(d::LogNormalExpert, yl, yu)
    return _zdensz_series((log(yl) - d.μ) / d.σ) - _zdensz_series((log(yu) - d.μ) / d.σ)
end

function _int_obs_logY_sq(d::LogNormalExpert, yl, yu, expert_ll)
    if yl == yu
        return (log(yl))^2
    else
        return exp(-expert_ll) * (
            ((d.σ)^2 + (d.μ)^2) * _diff_dist_series(d, yl, yu) +
            2.0 * d.σ * d.μ * invsqrt2π * _diff_dens_series(d, yl, yu) +
            (d.σ)^2 * invsqrt2π * _diff_zdensz_series(d, yl, yu)
        )
    end
end

function _int_lat_logY_sq(d::LogNormalExpert, tl, tu, expert_tn_bar)
    return exp(-expert_tn_bar) * (
        ((d.σ)^2 + (d.μ)^2) - (
            ((d.σ)^2 + (d.μ)^2) * _diff_dist_series(d, tl, tu) +
            2.0 * d.σ * d.μ * invsqrt2π * _diff_dens_series(d, tl, tu) +
            (d.σ)^2 * invsqrt2π * _diff_zdensz_series(d, tl, tu)
        )
    )
end

## EM: M-Step
function EM_M_expert(d::LogNormalExpert,
    tl, yl, yu, tu,
    exposure,
    z_e_obs, z_e_lat, k_e;
    penalty=true, pen_pararms_jk=[1.0 1.0])
    expert_ll_pos = expert_ll.(d, tl, yl, yu, tu)
    expert_tn_bar_pos = expert_tn_bar.(d, tl, yl, yu, tu)

    # Further E-Step
    logY_e_obs = vec(_int_obs_logY.(d, yl, yu, expert_ll_pos))
    logY_e_lat = vec(_int_lat_logY.(d, tl, tu, expert_tn_bar_pos))
    nan2num(logY_e_lat, 0.0) # get rid of NaN

    logY_sq_e_obs = vec(_int_obs_logY_sq.(d, yl, yu, expert_ll_pos))
    logY_sq_e_lat = vec(_int_lat_logY_sq.(d, tl, tu, expert_tn_bar_pos))
    nan2num(logY_sq_e_lat, 0.0) # get rid of NaN

    # Update parameters
    pos_idx = (yu .!= 0.0)
    term_zkz = z_e_obs[pos_idx] .+ (z_e_lat[pos_idx] .* k_e[pos_idx])
    term_zkz_logY =
        (z_e_obs[pos_idx] .* logY_e_obs[pos_idx]) .+
        (z_e_lat[pos_idx] .* k_e[pos_idx] .* logY_e_lat[pos_idx])
    term_zkz_logY_sq =
        (z_e_obs[pos_idx] .* logY_sq_e_obs[pos_idx]) .+
        (z_e_lat[pos_idx] .* k_e[pos_idx] .* logY_sq_e_lat[pos_idx])

    μ_new = sum(term_zkz_logY)[1] / sum(term_zkz)[1]

    demominator = penalty ? (sum(term_zkz)[1] + (pen_pararms_jk[2] - 1)) : sum(term_zkz)[1]
    numerator = if penalty
        (
            sum(term_zkz_logY_sq)[1] - 2.0 * μ_new * sum(term_zkz_logY)[1] +
            (μ_new)^2 * sum(term_zkz)[1] + (pen_pararms_jk[1] - 1)
        )
    else
        (
            sum(term_zkz_logY_sq)[1] - 2.0 * μ_new * sum(term_zkz_logY)[1] +
            (μ_new)^2 * sum(term_zkz)[1]
        )
    end
    tmp = numerator / demominator
    σ_new = sqrt(maximum([0.0, tmp]))

    return LogNormalExpert(μ_new, σ_new)
end

## EM: M-Step, exact observations
function EM_M_expert_exact(d::LogNormalExpert,
    ye, exposure,
    z_e_obs;
    penalty=true, pen_pararms_jk=[1.0 1.0])

    # Further E-Step
    logY_e_obs = log.(ye)

    logY_sq_e_obs = (log.(ye)) .^ 2

    # Update parameters
    pos_idx = (ye .!= 0.0)
    term_zkz = z_e_obs[pos_idx]
    term_zkz_logY = (z_e_obs[pos_idx] .* logY_e_obs[pos_idx])
    term_zkz_logY_sq = (z_e_obs[pos_idx] .* logY_sq_e_obs[pos_idx])

    μ_new = sum(term_zkz_logY)[1] / sum(term_zkz)[1]

    demominator = penalty ? (sum(term_zkz)[1] + (pen_pararms_jk[2] - 1)) : sum(term_zkz)[1]
    numerator = if penalty
        (
            sum(term_zkz_logY_sq)[1] - 2.0 * μ_new * sum(term_zkz_logY)[1] +
            (μ_new)^2 * sum(term_zkz)[1] + (pen_pararms_jk[1] - 1)
        )
    else
        (
            sum(term_zkz_logY_sq)[1] - 2.0 * μ_new * sum(term_zkz_logY)[1] +
            (μ_new)^2 * sum(term_zkz)[1]
        )
    end
    tmp = numerator / demominator
    σ_new = sqrt(maximum([0.0, tmp]))

    return LogNormalExpert(μ_new, σ_new)
end
