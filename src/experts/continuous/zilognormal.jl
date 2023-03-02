struct ZILogNormalExpert{T<:Real} <: ZIContinuousExpert
    p::T
    μ::T
    σ::T
    ZILogNormalExpert{T}(p::T, µ::T, σ::T) where {T<:Real} = new{T}(p, µ, σ)
end

function ZILogNormalExpert(p::T, μ::T, σ::T; check_args=true) where {T<:Real}
    check_args && @check_args(ZILogNormalExpert, 0 <= p <= 1 && σ >= zero(σ))
    return ZILogNormalExpert{T}(p, μ, σ)
end

#### Outer constructors
ZILogNormalExpert(p::Real, μ::Real, σ::Real) = ZILogNormalExpert(promote(p, μ, σ)...)
function ZILogNormalExpert(p::Integer, μ::Integer, σ::Integer)
    return ZILogNormalExpert(float(p), float(μ), float(σ))
end
ZILogNormalExpert() = ZILogNormalExpert(0.5, 0.0, 1.0)

## Conversion
function convert(::Type{ZILogNormalExpert{T}}, p::S, μ::S, σ::S) where {T<:Real,S<:Real}
    return ZILogNormalExpert(T(p), T(μ), T(σ))
end
function convert(
    ::Type{ZILogNormalExpert{T}}, d::ZILogNormalExpert{S}
) where {T<:Real,S<:Real}
    return ZILogNormalExpert(T(d.p), T(d.μ), T(d.σ); check_args=false)
end
copy(d::ZILogNormalExpert) = ZILogNormalExpert(d.p, d.μ, d.σ; check_args=false)

## Loglikelihood of Expoert
function logpdf(d::ZILogNormalExpert, x...)
    return Distributions.logpdf.(Distributions.LogNormal(d.μ, d.σ), x...)
end
function pdf(d::ZILogNormalExpert, x...)
    return Distributions.pdf.(Distributions.LogNormal(d.μ, d.σ), x...)
end
function logcdf(d::ZILogNormalExpert, x...)
    return Distributions.logcdf.(Distributions.LogNormal(d.μ, d.σ), x...)
end
function cdf(d::ZILogNormalExpert, x...)
    return Distributions.cdf.(Distributions.LogNormal(d.μ, d.σ), x...)
end

function expert_ll_exact(d::ZILogNormalExpert, x::Real)
    return (x == 0.0) ? log(p_zero(d)) : log(1 - p_zero(d)) + LRMoE.logpdf(d, x)
end
function expert_ll(d::ZILogNormalExpert, tl::Real, yl::Real, yu::Real, tu::Real)
    expert_ll_pos = LRMoE.expert_ll(LRMoE.LogNormalExpert(d.μ, d.σ), tl, yl, yu, tu)
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
function expert_tn(d::ZILogNormalExpert, tl::Real, yl::Real, yu::Real, tu::Real)
    expert_tn_pos = LRMoE.expert_tn(LRMoE.LogNormalExpert(d.μ, d.σ), tl, yl, yu, tu)
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
function expert_tn_bar(d::ZILogNormalExpert, tl::Real, yl::Real, yu::Real, tu::Real)
    expert_tn_bar_pos = LRMoE.expert_tn_bar(LRMoE.LogNormalExpert(d.μ, d.σ), tl, yl, yu, tu)
    # Deal with zero inflation
    p0 = p_zero(d)
    expert_tn_bar = if (tl > 0.0)
        log.(p0 + (1 - p0) * exp.(expert_tn_bar_pos))
    else
        log.(0.0 + (1 - p0) * exp.(expert_tn_bar_pos))
    end
    return expert_tn_bar
end

exposurize_expert(d::ZILogNormalExpert; exposure=1) = d

## Parameters
params(d::ZILogNormalExpert) = (d.p, d.μ, d.σ)
p_zero(d::ZILogNormalExpert) = d.p
function params_init(y, d::ZILogNormalExpert)
    p_init = sum(y .== 0.0) / sum(y .>= 0.0)
    pos_idx = (y .> 0.0)
    μ_init, σ_init = mean(log.(y[pos_idx])), sqrt(var(log.(y[pos_idx])))
    μ_init = isnan(μ_init) ? 0.0 : μ_init
    σ_init = isnan(σ_init) ? 1.0 : σ_init
    return ZILogNormalExpert(p_init, μ_init, σ_init)
end

## KS stats for parameter initialization
function ks_distance(y, d::ZILogNormalExpert)
    p_zero = sum(y .== 0.0) / sum(y .>= 0.0)
    return max(
        abs(p_zero - d.p),
        (1 - d.p) *
        HypothesisTests.ksstats(y[y .> 0.0], Distributions.LogNormal(d.μ, d.σ))[2],
    )
end

## Simululation
function sim_expert(d::ZILogNormalExpert)
    return (1 .- Distributions.rand(Distributions.Bernoulli(d.p), 1)[1]) .*
           Distributions.rand(Distributions.LogNormal(d.μ, d.σ), 1)[1]
end

## penalty
penalty_init(d::ZILogNormalExpert) = [2.0 2.0]
no_penalty_init(d::ZILogNormalExpert) = [1.0 1.0]
penalize(d::ZILogNormalExpert, p) = -0.5 * (p[1] - 1) / (d.σ * d.σ) - (p[2] - 1) * log(d.σ)

## statistics
mean(d::ZILogNormalExpert) = (1 - d.p) * mean(Distributions.LogNormal(d.μ, d.σ))
function var(d::ZILogNormalExpert)
    return (1 - d.p) * var(Distributions.LogNormal(d.μ, d.σ)) +
           d.p * (1 - d.p) * (mean(Distributions.LogNormal(d.μ, d.σ)))^2
end
function quantile(d::ZILogNormalExpert, p)
    return p <= d.p ? 0.0 : quantile(Distributions.LogNormal(d.μ, d.σ), p - d.p)
end
lev(d::ZILogNormalExpert, u) = (1 - d.p) * lev(LogNormalExpert(d.μ, d.σ), u)
excess(d::ZILogNormalExpert, u) = mean(d) - lev(d, u)

## EM: M-Step
function EM_M_expert(d::ZILogNormalExpert,
    tl, yl, yu, tu,
    exposure,
    z_e_obs, z_e_lat, k_e;
    penalty=true, pen_pararms_jk=[Inf 1.0 Inf])

    # Old parameters
    p_old = p_zero(d)

    # Update zero probability
    expert_ll_pos = expert_ll.(LRMoE.LogNormalExpert(d.μ, d.σ), tl, yl, yu, tu)
    expert_tn_bar_pos = expert_tn_bar.(LRMoE.LogNormalExpert(d.μ, d.σ), tl, yl, yu, tu)

    z_zero_e_obs = z_e_obs .* EM_E_z_zero_obs(yl, p_old, expert_ll_pos)
    z_pos_e_obs = z_e_obs .- z_zero_e_obs
    z_zero_e_lat = z_e_lat .* EM_E_z_zero_lat(tl, p_old, expert_tn_bar_pos)
    z_pos_e_lat = z_e_lat .- z_zero_e_lat
    p_new = EM_M_zero(z_zero_e_obs, z_pos_e_obs, z_zero_e_lat, z_pos_e_lat, k_e)

    p_new = max(0.0, min(1 - 1e-08, p_new))

    # Update parameters: call its positive part
    tmp_exp = LogNormalExpert(d.μ, d.σ)
    tmp_update = EM_M_expert(tmp_exp,
        tl, yl, yu, tu,
        exposure,
        z_pos_e_obs, z_pos_e_lat, k_e;
        penalty=penalty, pen_pararms_jk=pen_pararms_jk)

    return ZILogNormalExpert(p_new, tmp_update.μ, tmp_update.σ)
end

## EM: M-Step, exact observations
function EM_M_expert_exact(d::ZILogNormalExpert,
    ye, exposure,
    z_e_obs;
    penalty=true, pen_pararms_jk=[1.0 1.0])

    # Old parameters
    μ_old = d.μ
    σ_old = d.σ
    p_old = p_zero(d)

    # Update zero probability
    expert_ll_pos = expert_ll_exact.(LRMoE.LogNormalExpert(μ_old, σ_old), ye)

    z_zero_e_obs = z_e_obs .* EM_E_z_zero_obs(ye, p_old, expert_ll_pos)
    z_pos_e_obs = z_e_obs .- z_zero_e_obs
    p_new = EM_M_zero(z_zero_e_obs, z_pos_e_obs, 0.0, 0.0, 0.0)

    p_new = max(0.0, min(1 - 1e-08, p_new))

    # Update parameters: call its positive part
    tmp_exp = LogNormalExpert(μ_old, σ_old)
    tmp_update = EM_M_expert_exact(tmp_exp,
        ye,
        exposure,
        z_pos_e_obs;
        penalty=penalty, pen_pararms_jk=pen_pararms_jk)

    return ZILogNormalExpert(p_new, tmp_update.μ, tmp_update.σ)
end