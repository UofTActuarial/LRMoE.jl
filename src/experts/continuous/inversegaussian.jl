"""
    InverseGaussianExpert(μ, λ)

PDF:

```math
f(x; \\mu, \\lambda) = \\sqrt{\\frac{\\lambda}{2\\pi x^3}}
\\exp\\left(\\frac{-\\lambda(x-\\mu)^2}{2\\mu^2x}\\right), 
\\quad x > 0
```

See also: [Inverse Gaussian Distribution](https://en.wikipedia.org/wiki/Inverse_Gaussian_distribution) (Wikipedia) 

"""
struct InverseGaussianExpert{T<:Real} <: NonZIContinuousExpert
    μ::T
    λ::T
    InverseGaussianExpert{T}(µ::T, λ::T) where {T<:Real} = new{T}(µ, λ)
end

function InverseGaussianExpert(μ::T, λ::T; check_args=true) where {T<:Real}
    check_args && @check_args(InverseGaussianExpert, μ >= zero(μ) && λ > zero(λ))
    return InverseGaussianExpert{T}(μ, λ)
end

## Outer constructors
InverseGaussianExpert(μ::Real, λ::Real) = InverseGaussianExpert(promote(μ, λ)...)
InverseGaussianExpert(μ::Integer, λ::Integer) = InverseGaussianExpert(float(μ), float(λ))
InverseGaussianExpert() = InverseGaussianExpert(1.0, 1.0)

## Conversion
function convert(::Type{InverseGaussianExpert{T}}, μ::S, λ::S) where {T<:Real,S<:Real}
    return InverseGaussianExpert(T(μ), T(λ))
end
function convert(
    ::Type{InverseGaussianExpert{T}}, d::InverseGaussianExpert{S}
) where {T<:Real,S<:Real}
    return InverseGaussianExpert(T(d.μ), T(d.λ); check_args=false)
end
copy(d::InverseGaussianExpert) = InverseGaussianExpert(d.μ, d.λ; check_args=false)

## Loglikelihood of Expert
function logpdf(d::InverseGaussianExpert, x...)
    return if isinf(x...)
        -Inf
    else
        Distributions.logpdf.(Distributions.InverseGaussian(d.μ, d.λ), x...)
    end
end
function pdf(d::InverseGaussianExpert, x...)
    return if isinf(x...)
        0.0
    else
        Distributions.pdf.(Distributions.InverseGaussian(d.μ, d.λ), x...)
    end
end
function logcdf(d::InverseGaussianExpert, x...)
    return if isinf(x...)
        0.0
    else
        Distributions.logcdf.(Distributions.InverseGaussian(d.μ, d.λ), x...)
    end
end
function cdf(d::InverseGaussianExpert, x...)
    return if isinf(x...)
        1.0
    else
        Distributions.cdf.(Distributions.InverseGaussian(d.μ, d.λ), x...)
    end
end

## expert_ll, etc
expert_ll_exact(d::InverseGaussianExpert, x::Real) = LRMoE.logpdf(d, x)
function expert_ll(d::InverseGaussianExpert, tl::Real, yl::Real, yu::Real, tu::Real)
    expert_ll = if (yl == yu)
        logpdf.(d, yl)
    else
        logcdf.(d, yu) + log1mexp.(logcdf.(d, yl) - logcdf.(d, yu))
    end
    expert_ll = (tu == 0.0) ? -Inf : expert_ll
    return expert_ll
end
function expert_tn(d::InverseGaussianExpert, tl::Real, yl::Real, yu::Real, tu::Real)
    expert_tn = if (tl == tu)
        logpdf.(d, tl)
    else
        logcdf.(d, tu) + log1mexp.(logcdf.(d, tl) - logcdf.(d, tu))
    end
    expert_tn = (tu == 0.0) ? -Inf : expert_tn
    return expert_tn
end
function expert_tn_bar(d::InverseGaussianExpert, tl::Real, yl::Real, yu::Real, tu::Real)
    expert_tn_bar = if (tl == tu)
        0.0
    else
        log1mexp.(logcdf.(d, tu) + log1mexp.(logcdf.(d, tl) - logcdf.(d, tu)))
    end
    return expert_tn_bar
end

exposurize_expert(d::InverseGaussianExpert; exposure=1) = d

## Parameters
params(d::InverseGaussianExpert) = (d.μ, d.λ)
function params_init(y, d::InverseGaussianExpert)
    pos_idx = (y .> 0.0)
    μ, σ2 = mean(y[pos_idx]), var(y[pos_idx])
    μ_init = μ
    λ_init = μ^3 / σ2
    if isnan(μ_init) || isnan(λ_init)
        return InverseGaussianExpert()
    else
        return InverseGaussianExpert(μ_init, λ_init)
    end
end

## KS stats for parameter initialization
function ks_distance(y, d::InverseGaussianExpert)
    p_zero = sum(y .== 0.0) / sum(y .>= 0.0)
    return max(
        abs(p_zero - 0.0),
        (1 - 0.0) *
        HypothesisTests.ksstats(y[y .> 0.0], Distributions.InverseGaussian(d.μ, d.λ))[2],
    )
end

## Simululation
function sim_expert(d::InverseGaussianExpert)
    return Distributions.rand(Distributions.InverseGaussian(d.μ, d.λ), 1)[1]
end

## penalty
penalty_init(d::InverseGaussianExpert) = [1.0 Inf 1.0 Inf]
no_penalty_init(d::InverseGaussianExpert) = [1.0 Inf 1.0 Inf]
function penalize(d::InverseGaussianExpert, p)
    return (p[1] - 1) * log(d.μ) - d.μ / p[2] + (p[3] - 1) * log(d.λ) - d.λ / p[4]
end

## statistics
mean(d::InverseGaussianExpert) = mean(Distributions.InverseGaussian(d.μ, d.λ))
var(d::InverseGaussianExpert) = var(Distributions.InverseGaussian(d.μ, d.λ))
quantile(d::InverseGaussianExpert, p) = quantile(Distributions.InverseGaussian(d.μ, d.λ), p)
function lev(d::InverseGaussianExpert, u)
    if isinf(u)
        return mean(d)
    else
        z = (u - d.μ) / d.μ
        y = (u + d.μ) / d.μ
        return u - d.μ * z * cdf.(Normal(0, 1), z * (d.λ / u)^0.5) -
               d.μ * y * exp(2 * d.λ / d.μ) * cdf.(Normal(0, 1), -y * (d.λ / u)^0.5)
    end
end
excess(d::InverseGaussianExpert, u) = mean(d) - lev(d, u)

## Misc functions for E-Step
function _int_y_func(d::InverseGaussianExpert, x)
    return (iszero(x) || isinf(x)) ? 0.0 : x * pdf.(d, x)
end

function _int_obs_Y_raw(d::InverseGaussianExpert, yl, yu)
    if yl == yu
        return yl * pdf.(d, yl)
    else
        return quadgk.(x -> _int_y_func(d, x), yl, yu, rtol=1e-8)[1]
    end
end

function _int_obs_Y_raw_threaded(d::InverseGaussianExpert, yl, yu)
    result = fill(NaN, length(yl))
    @threads for i in 1:length(yl)
        result[i] = _int_obs_Y_raw(d, yl[i], yu[i])
    end
    return result
end

function _int_lat_Y_raw(d::InverseGaussianExpert, tl, tu)
    return (tl == 0 ? 0.0 : quadgk.(x -> _int_y_func(d, x), 0.0, tl, rtol=1e-8)[1]) +
           (isinf(tu) ? 0.0 : quadgk.(x -> _int_y_func(d, x), tu, Inf, rtol=1e-8)[1])
end

function _int_lat_Y_raw_threaded(d::InverseGaussianExpert, tl, tu)
    result = fill(NaN, length(tl))
    @threads for i in 1:length(tl)
        result[i] = _int_lat_Y_raw(d, tl[i], tu[i])
    end
    return result
end

function _int_logy_func(d::InverseGaussianExpert, x)
    return (iszero(x) || isinf(x)) ? 0.0 : log(x) * pdf.(d, x)
end

function _int_obs_logY_raw(d::InverseGaussianExpert, yl, yu)
    if yl == yu
        return log(yl) * pdf.(d, yl)
    else
        return quadgk.(x -> _int_logy_func(d, x), yl, yu, rtol=1e-8)[1]
    end
end

function _int_obs_logY_raw_threaded(d::InverseGaussianExpert, yl, yu)
    result = fill(NaN, length(yl))
    @threads for i in 1:length(yl)
        result[i] = _int_obs_logY_raw(d, yl[i], yu[i])
    end
    return result
end

function _int_lat_logY_raw(d::InverseGaussianExpert, tl, tu)
    return (tl == 0 ? 0.0 : quadgk.(x -> _int_logy_func(d, x), 0.0, tl, rtol=1e-8)[1]) +
           (isinf(tu) ? 0.0 : quadgk.(x -> _int_logy_func(d, x), tu, Inf, rtol=1e-8)[1])
end

function _int_lat_logY_raw_threaded(d::InverseGaussianExpert, tl, tu)
    result = fill(NaN, length(tl))
    @threads for i in 1:length(tl)
        result[i] = _int_lat_logY_raw(d, tl[i], tu[i])
    end
    return result
end

function _int_invy_func(d::InverseGaussianExpert, x)
    return (iszero(x) || isinf(x)) ? 0.0 : 1 / (x) * pdf.(d, x)
end

function _int_obs_invY_raw(d::InverseGaussianExpert, yl, yu)
    if yl == yu
        return 1 / yl * pdf.(d, yl)
    else
        return quadgk.(x -> _int_invy_func(d, x), yl, yu, rtol=1e-8)[1]
    end
end

function _int_obs_invY_raw_threaded(d::InverseGaussianExpert, yl, yu)
    result = fill(NaN, length(yl))
    @threads for i in 1:length(yl)
        result[i] = _int_obs_invY_raw(d, yl[i], yu[i])
    end
    return result
end

function _int_lat_invY_raw(d::InverseGaussianExpert, tl, tu)
    return (tl == 0 ? 0.0 : quadgk.(x -> _int_invy_func(d, x), 0.0, tl, rtol=1e-8)[1]) +
           (isinf(tu) ? 0.0 : quadgk.(x -> _int_invy_func(d, x), tu, Inf, rtol=1e-8)[1])
end

function _int_lat_invY_raw_threaded(d::InverseGaussianExpert, tl, tu)
    result = fill(NaN, length(tl))
    @threads for i in 1:length(tl)
        result[i] = _int_lat_invY_raw(d, tl[i], tu[i])
    end
    return result
end

## EM: M-Step
function EM_M_expert(d::InverseGaussianExpert,
    tl, yl, yu, tu,
    exposure,
    z_e_obs, z_e_lat, k_e;
    penalty=true, pen_pararms_jk=[1.0 Inf 1.0 Inf])
    if d.μ < 0.000001 || d.λ < 0.000001
        return d
    end

    expert_ll_pos = expert_ll.(d, tl, yl, yu, tu)
    expert_tn_bar_pos = expert_tn_bar.(d, tl, yl, yu, tu)

    # Further E-Step
    yl_yu_unique = unique_bounds(yl, yu)

    int_obs_Y_tmp = _int_obs_Y_raw_threaded(d, yl_yu_unique[:, 1], yl_yu_unique[:, 2])
    # int_obs_logY_tmp = _int_obs_logY_raw.(d, yl_yu_unique[:,1], yl_yu_unique[:,2])
    int_obs_invY_tmp = _int_obs_invY_raw_threaded(d, yl_yu_unique[:, 1], yl_yu_unique[:, 2])

    Y_e_obs =
        exp.(-expert_ll_pos) .*
        int_obs_Y_tmp[match_unique_bounds_threaded(hcat(vec(yl), vec(yu)), yl_yu_unique)]
    # logY_e_obs = exp.(-expert_ll_pos) .* int_obs_logY_tmp[match_unique_bounds(hcat(vec(yl), vec(yu)), yl_yu_unique)]
    invY_e_obs =
        exp.(-expert_ll_pos) .*
        int_obs_invY_tmp[match_unique_bounds_threaded(hcat(vec(yl), vec(yu)), yl_yu_unique)]

    nan2num(Y_e_obs, 0.0) # get rid of NaN
    # nan2num(logY_e_obs, 0.0) # get rid of NaN
    nan2num(invY_e_obs, 0.0) # get rid of NaN

    tl_tu_unique = unique_bounds(tl, tu)

    int_lat_Y_tmp = _int_lat_Y_raw_threaded(d, tl_tu_unique[:, 1], tl_tu_unique[:, 2])
    # int_lat_logY_tmp = _int_lat_logY_raw.(d, tl_tu_unique[:,1], tl_tu_unique[:,2])
    int_lat_invY_tmp = _int_lat_invY_raw_threaded(d, tl_tu_unique[:, 1], tl_tu_unique[:, 2])

    Y_e_lat =
        exp.(-expert_tn_bar_pos) .*
        int_lat_Y_tmp[match_unique_bounds_threaded(hcat(vec(tl), vec(tu)), tl_tu_unique)]
    # logY_e_lat = exp.(-expert_tn_bar_pos) .* int_lat_logY_tmp[match_unique_bounds(hcat(vec(tl), vec(tu)), tl_tu_unique)]
    invY_e_lat =
        exp.(-expert_tn_bar_pos) .*
        int_lat_invY_tmp[match_unique_bounds_threaded(hcat(vec(tl), vec(tu)), tl_tu_unique)]

    nan2num(Y_e_lat, 0.0) # get rid of NaN
    # nan2num(logY_e_lat, 0.0) # get rid of NaN
    nan2num(invY_e_lat, 0.0) # get rid of NaN

    # Update parameters
    pos_idx = (yu .!= 0.0)
    term_zkz = z_e_obs[pos_idx] .+ (z_e_lat[pos_idx] .* k_e[pos_idx])
    term_zkz_Y =
        (z_e_obs[pos_idx] .* Y_e_obs[pos_idx]) .+
        (z_e_lat[pos_idx] .* k_e[pos_idx] .* Y_e_lat[pos_idx])
    # term_zkz_logY = (z_e_obs[pos_idx] .* logY_e_obs[pos_idx]) .+ (z_e_lat[pos_idx] .* k_e[pos_idx] .* logY_e_lat[pos_idx])
    term_zkz_invY =
        (z_e_obs[pos_idx] .* invY_e_obs[pos_idx]) .+
        (z_e_lat[pos_idx] .* k_e[pos_idx] .* invY_e_lat[pos_idx])

    μ_new = sum(term_zkz_Y)[1] / sum(term_zkz)[1]
    λ_new =
        sum(term_zkz)[1] / (
            sum(term_zkz_Y)[1] / (μ_new)^2 - 2 * sum(term_zkz)[1] / μ_new +
            sum(term_zkz_invY)[1]
        )

    return InverseGaussianExpert(μ_new, λ_new)
end

## EM: M-Step, exact observations
function EM_M_expert_exact(d::InverseGaussianExpert,
    ye, exposure,
    z_e_obs;
    penalty=true, pen_pararms_jk=[1.0 Inf 1.0 Inf])
    if d.μ < 0.000001 || d.λ < 0.000001
        return d
    end

    # Further E-Step
    Y_e_obs = ye
    invY_e_obs = 1.0 ./ ye

    nan2num(Y_e_obs, 0.0) # get rid of NaN

    # Update parameters
    pos_idx = (ye .!= 0.0)
    term_zkz = z_e_obs[pos_idx]
    term_zkz_Y = z_e_obs[pos_idx] .* Y_e_obs[pos_idx]
    term_zkz_invY = z_e_obs[pos_idx] .* invY_e_obs[pos_idx]

    μ_new = sum(term_zkz_Y)[1] / sum(term_zkz)[1]
    λ_new =
        sum(term_zkz)[1] / (
            sum(term_zkz_Y)[1] / (μ_new)^2 - 2 * sum(term_zkz)[1] / μ_new +
            sum(term_zkz_invY)[1]
        )

    return InverseGaussianExpert(μ_new, λ_new)
end
