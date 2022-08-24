"""
    PoissonExpert(λ)

PMF:

```math
P(X = k) = \\frac{\\lambda^k}{k!} e^{-\\lambda}, \\quad \\text{ for } k = 0,1,2,\\ldots.
```

See also: [Poisson Distribution](https://en.wikipedia.org/wiki/Poisson_distribution) (Wikipedia) 

"""
struct PoissonExpert{T<:Real} <: NonZIDiscreteExpert
    λ::T
    PoissonExpert{T}(λ::T) where {T<:Real} = new{T}(λ)
end

function PoissonExpert(λ::T; check_args=true) where {T<:Real}
    check_args && @check_args(PoissonExpert, λ >= zero(λ))
    return PoissonExpert{T}(λ)
end

## Outer constructors
# PoissonExpert(λ::Real) = PoissonExpert(promote(λ)...)
PoissonExpert(λ::Integer) = PoissonExpert(float(λ))
PoissonExpert() = PoissonExpert(1.0)

## Conversion
function convert(::Type{PoissonExpert{T}}, λ::S) where {T<:Real,S<:Real}
    return PoissonExpert(T(λ))
end
function convert(::Type{PoissonExpert{T}}, d::PoissonExpert{S}) where {T<:Real,S<:Real}
    return PoissonExpert(T(d.λ); check_args=false)
end
copy(d::PoissonExpert) = PoissonExpert(d.λ; check_args=false)

## Loglikelihood of Expoert
function logpdf(d::PoissonExpert, x...)
    return isinf(x...) ? 0.0 : Distributions.logpdf.(Distributions.Poisson(d.λ), x...)
end
function pdf(d::PoissonExpert, x...)
    return isinf(x...) ? -Inf : Distributions.pdf.(Distributions.Poisson(d.λ), x...)
end
function logcdf(d::PoissonExpert, x...)
    return isinf(x...) ? 0.0 : Distributions.logcdf.(Distributions.Poisson(d.λ), x...)
end
function cdf(d::PoissonExpert, x...)
    return isinf(x...) ? 1.0 : Distributions.cdf.(Distributions.Poisson(d.λ), x...)
end

## expert_ll, etc
expert_ll_exact(d::PoissonExpert, x::Real) = LRMoE.logpdf(LRMoE.PoissonExpert(d.λ), x)
function expert_ll(d::PoissonExpert, tl::Real, yl::Real, yu::Real, tu::Real)
    # d_exp = LRMoE.PoissonExpert(d.λ*exposure)
    expert_ll = if (yl == yu)
        logpdf.(d, yl)
    else
        logcdf.(d, yu) + log1mexp.(logcdf.(d, ceil.(yl) .- 1) - logcdf.(d, yu))
    end
    return expert_ll
end
function expert_tn(d::PoissonExpert, tl::Real, yl::Real, yu::Real, tu::Real)
    # d_exp = LRMoE.PoissonExpert(d.λ*exposure)
    expert_tn = if (tl == tu)
        logpdf.(d, tl)
    else
        logcdf.(d, tu) + log1mexp.(logcdf.(d, ceil.(tl) .- 1) - logcdf.(d, tu))
    end
    return expert_tn
end
function expert_tn_bar(d::PoissonExpert, tl::Real, yl::Real, yu::Real, tu::Real)
    # d_exp = LRMoE.PoissonExpert(d.λ*exposure)
    expert_tn_bar = if (tl == tu)
        log1mexp.(logpdf.(d, tl))
    else
        log1mexp.(logcdf.(d, tu) + log1mexp.(logcdf.(d, ceil.(tl) .- 1) - logcdf.(d, tu)))
    end
    return expert_tn_bar
end

exposurize_expert(d::PoissonExpert; exposure=1) = PoissonExpert(d.λ * exposure)

## Parameters
params(d::PoissonExpert) = (d.λ)
function params_init(y, d::PoissonExpert)
    λ_init = mean(y)
    try
        return PoissonExpert(λ_init)
    catch
        PoissonExpert()
    end
end

## KS stats for parameter initialization
function ks_distance(y, d::PoissonExpert)
    p_zero = sum(y .== 0.0) / sum(y .>= 0.0)
    return max(
        abs(p_zero - 0.0),
        (1 - 0.0) * HypothesisTests.ksstats(y[y .> 0.0], Distributions.Poisson(d.λ))[2],
    )
end

## Simululation
sim_expert(d::PoissonExpert) = Distributions.rand(Distributions.Poisson(d.λ), 1)[1]

## penalty
penalty_init(d::PoissonExpert) = [2.0 1.0]
no_penalty_init(d::PoissonExpert) = [1.0 Inf]
penalize(d::PoissonExpert, p) = (p[1] - 1) * log(d.λ) - d.λ / p[2]

## statistics
mean(d::PoissonExpert) = mean(Distributions.Poisson(d.λ))
var(d::PoissonExpert) = var(Distributions.Poisson(d.λ))
quantile(d::PoissonExpert, p) = quantile(Distributions.Poisson(d.λ), p)

## Misc functions for E-Step

function _sum_densy_series(d::PoissonExpert, yl, yu)
    if isinf(yu)
        series = 0:(max(yl - 1, 0))
        return d.λ - sum(pdf.(d, series) .* series)[1]
    else
        series = yl:yu
        return sum(pdf.(d, series) .* series)[1]
    end
end

function _int_obs_Y_raw(d::PoissonExpert, yl, yu)
    # if yl == yu
    #     return yl
    # else
    #     return _sum_densy_series(d, yl, yu)
    # end
    return _sum_densy_series(d, yl, yu)
end

function _int_lat_Y_raw(d::PoissonExpert, tl, tu)
    return maximum([d.λ - _sum_densy_series(d, tl, tu), 0])
end

## EM: M-Step
function EM_M_expert(d::PoissonExpert,
    tl, yl, yu, tu,
    exposure,
    z_e_obs, z_e_lat, k_e;
    penalty=true, pen_pararms_jk=[2.0 1.0])

    # Old parameters
    λ_old = d.λ

    # Further E-Step
    # yl_yu_unique = unique_bounds(yl, yu)
    # int_obs_Y_tmp = _int_obs_Y_raw.(d, yl_yu_unique[:,1], yl_yu_unique[:,2])
    # Y_e_obs = exp.(-expert_ll_pos) .* int_obs_Y_tmp[match_unique_bounds(hcat(vec(yl), vec(yu)), yl_yu_unique)]
    # nan2num(Y_e_obs, 0.0) # get rid of NaN

    # tl_tu_unique = unique_bounds(tl, tu)
    # int_lat_Y_tmp = _int_lat_Y_raw.(d, tl_tu_unique[:,1], tl_tu_unique[:,2])
    # Y_e_lat = exp.(-expert_tn_bar_pos) .* int_lat_Y_tmp[match_unique_bounds(hcat(vec(tl), vec(tu)), tl_tu_unique)]
    # nan2num(Y_e_lat, 0.0) # get rid of NaN

    Y_e_obs = fill(0.0, length(yl))
    Y_e_lat = fill(0.0, length(yl))

    for i in 1:length(yl)
        d_expo = exposurize_expert(d; exposure=exposure[i])
        expert_ll_pos = expert_ll.(d_expo, tl[i], yl[i], yu[i], tu[i])
        expert_tn_bar_pos = expert_tn_bar.(d_expo, tl[i], yl[i], yu[i], tu[i])
        Y_e_obs[i] = exp(-expert_ll_pos) * _int_obs_Y_raw(d_expo, yl[i], yu[i])
        Y_e_lat[i] = exp(-expert_tn_bar_pos) * _int_lat_Y_raw(d_expo, tl[i], tu[i])
    end

    nan2num(Y_e_obs, 0.0)
    nan2num(Y_e_lat, 0.0)

    # Update parameters
    term_zkz = (z_e_obs .* exposure) .+ (z_e_lat .* k_e .* exposure)
    term_zkz_Y = (z_e_obs .* Y_e_obs) .+ (z_e_lat .* k_e .* Y_e_lat)

    λ_new = if penalty
        (
        (sum(term_zkz_Y)[1] - (pen_pararms_jk[1] - 1)) /
        (sum(term_zkz)[1] + 1 / pen_pararms_jk[2])
    )
    else
        (sum(term_zkz_Y)[1] / sum(term_zkz)[1])
    end

    # Need to deal with zero mass: λ is very small
    if λ_new < 0.00001
        λ_new = λ_old
    end

    return PoissonExpert(λ_new)
end

## EM: M-Step, exact observations
function EM_M_expert_exact(d::PoissonExpert,
    ye, exposure,
    # expert_ll_pos,
    z_e_obs;
    penalty=true, pen_pararms_jk=[Inf 1.0 Inf])

    # Old parameters
    λ_old = d.λ

    # Update parameters
    term_zkz = z_e_obs .* exposure
    term_zkz_Y = (z_e_obs .* ye)

    λ_new = if penalty
        (
        (sum(term_zkz_Y)[1] - (pen_pararms_jk[1] - 1)) /
        (sum(term_zkz)[1] + 1 / pen_pararms_jk[2])
    )
    else
        (sum(term_zkz_Y)[1] / sum(term_zkz)[1])
    end

    # Need to deal with zero mass: λ is very small
    if λ_new < 0.00001
        λ_new = λ_old
    end

    return PoissonExpert(λ_new)
end