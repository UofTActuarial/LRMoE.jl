struct ZIWeibullExpert{T<:Real} <: ZIContinuousExpert
    p::T
    k::T
    θ::T
    ZIWeibullExpert{T}(p::T, k::T, θ::T) where {T<:Real} = new{T}(p, k, θ)
end

function ZIWeibullExpert(p::T, k::T, θ::T; check_args=true) where {T<:Real}
    check_args && @check_args(ZIWeibullExpert, 0 <= p <= 1 && k >= one(k) && θ > zero(θ))
    return ZIWeibullExpert{T}(p, k, θ)
end

#### Outer constructors
ZIWeibullExpert(p::Real, k::Real, θ::Real) = ZIWeibullExpert(promote(p, k, θ)...)
function ZIWeibullExpert(p::Integer, k::Integer, θ::Integer)
    return ZIWeibullExpert(float(p), float(k), float(θ))
end
ZIWeibullExpert() = ZIWeibullExpert(0.5, 2.0, 1.0)

## Conversion
function convert(::Type{ZIWeibullExpert{T}}, p::S, k::S, θ::S) where {T<:Real,S<:Real}
    return ZIWeibullExpert(T(p), T(k), T(θ))
end
function convert(::Type{ZIWeibullExpert{T}}, d::ZIWeibullExpert{S}) where {T<:Real,S<:Real}
    return ZIWeibullExpert(T(d.p), T(d.k), T(d.θ); check_args=false)
end
copy(d::ZIWeibullExpert) = ZIWeibullExpert(d.p, d.k, d.θ; check_args=false)

## Loglikelihood of Expoert
function logpdf(d::ZIWeibullExpert, x...)
    return Distributions.logpdf.(Distributions.Weibull(d.k, d.θ), x...)
end
pdf(d::ZIWeibullExpert, x...) = Distributions.pdf.(Distributions.Weibull(d.k, d.θ), x...)
function logcdf(d::ZIWeibullExpert, x...)
    return Distributions.logcdf.(Distributions.Weibull(d.k, d.θ), x...)
end
cdf(d::ZIWeibullExpert, x...) = Distributions.cdf.(Distributions.Weibull(d.k, d.θ), x...)

function expert_ll_exact(d::ZIWeibullExpert, x::Real)
    return (x == 0.0) ? log(p_zero(d)) : log(1 - p_zero(d)) + LRMoE.logpdf(d, x)
end
function expert_ll(d::ZIWeibullExpert, tl::Real, yl::Real, yu::Real, tu::Real)
    expert_ll_pos = LRMoE.expert_ll(LRMoE.WeibullExpert(d.k, d.θ), tl, yl, yu, tu)
    # Deal with zero inflation
    p0 = p_zero(d)
    expert_ll = if (yl == 0.0)
        log.(p0 + (1 - p0) * exp.(expert_ll_pos))
    else
        log.(0.0 + (1 - p0) * exp.(expert_ll_pos))
    end
    expert_ll = (tu == 0.0) ? log.(p0) : expert_ll
    return expert_ll
end
function expert_tn(d::ZIWeibullExpert, tl::Real, yl::Real, yu::Real, tu::Real)
    expert_tn_pos = LRMoE.expert_tn(LRMoE.WeibullExpert(d.k, d.θ), tl, yl, yu, tu)
    # Deal with zero inflation
    p0 = p_zero(d)
    expert_tn = if (tl == 0.0)
        log.(p0 + (1 - p0) * exp.(expert_tn_pos))
    else
        log.(0.0 + (1 - p0) * exp.(expert_tn_pos))
    end
    expert_tn = (tu == 0.0) ? log.(p0) : expert_tn
    return expert_tn
end
function expert_tn_bar(d::ZIWeibullExpert, tl::Real, yl::Real, yu::Real, tu::Real)
    expert_tn_bar_pos = LRMoE.expert_tn_bar(LRMoE.WeibullExpert(d.k, d.θ), tl, yl, yu, tu)
    # Deal with zero inflation
    p0 = p_zero(d)
    expert_tn_bar = if (tl > 0.0)
        log.(p0 + (1 - p0) * exp.(expert_tn_bar_pos))
    else
        log.(0.0 + (1 - p0) * exp.(expert_tn_bar_pos))
    end
    return expert_tn_bar
end

exposurize_expert(d::ZIWeibullExpert; exposure=1) = d

## Parameters
params(d::ZIWeibullExpert) = (d.p, d.k, d.θ)
p_zero(d::ZIWeibullExpert) = d.p
function params_init(y, d::ZIWeibullExpert)
    p_init = sum(y .== 0.0) / sum(y .>= 0.0)
    pos_idx = (y .> 0.0)

    k_init, θ_init = params(params_init(y[pos_idx], WeibullExpert()))
    try
        return ZIWeibullExpert(p_init, k_init, θ_init)
    catch
        ZIWeibullExpert()
    end
end

## KS stats for parameter initialization
function ks_distance(y, d::ZIWeibullExpert)
    p_zero = sum(y .== 0.0) / sum(y .>= 0.0)
    return max(
        abs(p_zero - d.p),
        (1 - d.p) *
        HypothesisTests.ksstats(y[y .> 0.0], Distributions.Weibull(d.k, d.θ))[2],
    )
end

## Simululation
function sim_expert(d::ZIWeibullExpert)
    return (1 .- Distributions.rand(Distributions.Bernoulli(d.p), 1)[1]) .*
           Distributions.rand(Distributions.Weibull(d.k, d.θ), 1)[1]
end

## penalty
penalty_init(d::ZIWeibullExpert) = [2.0 10.0 2.0 10.0]
no_penalty_init(d::ZIWeibullExpert) = [1.0 Inf 1.0 Inf]
function penalize(d::ZIWeibullExpert, p)
    return (p[1] - 1) * log(d.k) - d.k / p[2] + (p[3] - 1) * log(d.θ) - d.θ / p[4]
end

## statistics
mean(d::ZIWeibullExpert) = (1 - d.p) * mean(Distributions.Weibull(d.k, d.θ))
function var(d::ZIWeibullExpert)
    return (1 - d.p) * var(Distributions.Weibull(d.k, d.θ)) +
           d.p * (1 - d.p) * (mean(Distributions.Weibull(d.k, d.θ)))^2
end
function quantile(d::ZIWeibullExpert, p)
    return p <= d.p ? 0.0 : quantile(Distributions.Weibull(d.k, d.θ), p - d.p)
end
lev(d::ZIWeibullExpert, u) = (1 - d.p) * lev(WeibullExpert(d.k, d.θ), u)
excess(d::ZIWeibullExpert, u) = mean(d) - lev(d, u)

## EM: M-Step
function EM_M_expert(d::ZIWeibullExpert,
    tl, yl, yu, tu,
    exposure,
    z_e_obs, z_e_lat, k_e;
    penalty=true, pen_pararms_jk=[1.0 Inf 1.0 Inf])

    # Old parameters
    p_old = p_zero(d)

    if p_old > 0.999999
        return d
    end

    # Update zero probability
    expert_ll_pos = expert_ll.(LRMoE.WeibullExpert(d.k, d.θ), tl, yl, yu, tu)
    expert_tn_bar_pos = expert_tn_bar.(LRMoE.WeibullExpert(d.k, d.θ), tl, yl, yu, tu)

    z_zero_e_obs = z_e_obs .* EM_E_z_zero_obs(yl, p_old, expert_ll_pos)
    z_pos_e_obs = z_e_obs .- z_zero_e_obs
    z_zero_e_lat = z_e_lat .* EM_E_z_zero_lat(tl, p_old, expert_tn_bar_pos)
    z_pos_e_lat = z_e_lat .- z_zero_e_lat
    p_new = EM_M_zero(z_zero_e_obs, z_pos_e_obs, z_zero_e_lat, z_pos_e_lat, k_e)

    # Update parameters: call its positive part
    tmp_exp = WeibullExpert(d.k, d.θ)
    tmp_update = EM_M_expert(tmp_exp,
        tl, yl, yu, tu,
        exposure,
        z_pos_e_obs, z_pos_e_lat, k_e;
        penalty=penalty, pen_pararms_jk=pen_pararms_jk)

    return ZIWeibullExpert(p_new, tmp_update.k, tmp_update.θ)
end

## EM: M-Step, exact observations
function EM_M_expert_exact(d::ZIWeibullExpert,
    ye, exposure,
    z_e_obs;
    penalty=true, pen_pararms_jk=[Inf 1.0 Inf])

    # Old parameters
    p_old = p_zero(d)

    if p_old > 0.999999
        return d
    end

    # Update zero probability
    expert_ll_pos = expert_ll_exact.(LRMoE.WeibullExpert(d.k, d.θ), ye)

    z_zero_e_obs = z_e_obs .* EM_E_z_zero_obs(ye, p_old, expert_ll_pos)
    z_pos_e_obs = z_e_obs .- z_zero_e_obs
    p_new = EM_M_zero(z_zero_e_obs, z_pos_e_obs, 0.0, 0.0, 0.0)

    # Update parameters: call its positive part
    tmp_exp = WeibullExpert(d.k, d.θ)
    tmp_update = EM_M_expert_exact(tmp_exp,
        ye, exposure,
        z_pos_e_obs;
        penalty=penalty, pen_pararms_jk=pen_pararms_jk)

    return ZIWeibullExpert(p_new, tmp_update.k, tmp_update.θ)
end