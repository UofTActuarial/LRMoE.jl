struct ZIGammaCountExpert{T<:Real} <: ZIDiscreteExpert
    p::T
    m::T
    s::T
    ZIGammaCountExpert{T}(p, m, s) where {T<:Real} = new{T}(p, m, s)
end

function ZIGammaCountExpert(p::T, m::T, s::T; check_args=true) where {T<:Real}
    check_args && @check_args(ZIGammaCountExpert, 0 <= p <= 1 && m > zero(m) && s > zero(s))
    return ZIGammaCountExpert{T}(p, m, s)
end

## Outer constructors
ZIGammaCountExpert(p::Real, m::Real, s::Real) = ZIGammaCountExpert(promote(p, m, s)...)
function ZIGammaCountExpert(p::Integer, m::Integer, s::Integer)
    return ZIGammaCountExpert(float(p), float(m), float(s))
end
ZIGammaCountExpert() = ZIGammaCountExpert(0.50, 2.0, 1.0)

## Conversion
function convert(::Type{ZIGammaCountExpert{T}}, n::S, p::S) where {T<:Real,S<:Real}
    return ZIGammaCountExpert(T(p), T(m), T(s))
end
function convert(
    ::Type{ZIGammaCountExpert{T}}, d::ZIGammaCountExpert{S}
) where {T<:Real,S<:Real}
    return ZIGammaCountExpert(T(d.p), T(d.m), T(d.s); check_args=false)
end
copy(d::ZIGammaCountExpert) = ZIGammaCountExpert(d.p, d.m, d.s; check_args=false)

## Loglikelihood of Expoert
function logpdf(d::ZIGammaCountExpert, x...)
    return isinf(x...) ? 0.0 : Distributions.logpdf.(LRMoE.GammaCount(d.m, d.s), x...)
end
function pdf(d::ZIGammaCountExpert, x...)
    return isinf(x...) ? -Inf : Distributions.pdf.(LRMoE.GammaCount(d.m, d.s), x...)
end
function logcdf(d::ZIGammaCountExpert, x...)
    return isinf(x...) ? 0.0 : Distributions.logcdf.(LRMoE.GammaCount(d.m, d.s), x...)
end
function cdf(d::ZIGammaCountExpert, x...)
    return isinf(x...) ? 1.0 : Distributions.cdf.(LRMoE.GammaCount(d.m, d.s), x...)
end

## expert_ll, etc
function expert_ll_exact(d::ZIGammaCountExpert, x::Real)
    return if (x == 0.0)
        log.(
        p_zero(d) +
        (1 - p_zero(d)) * exp.(LRMoE.logpdf(LRMoE.GammaCountExpert(d.m, d.s), x))
    )
    else
        log.(1 - p_zero(d)) + LRMoE.logpdf(LRMoE.GammaCountExpert(d.m, d.s), x)
    end
end
function expert_ll(d::ZIGammaCountExpert, tl::Real, yl::Real, yu::Real, tu::Real)
    expert_ll_pos = LRMoE.expert_ll(LRMoE.GammaCountExpert(d.m, d.s), tl, yl, yu, tu)
    # Deal with zero inflation
    p0 = p_zero(d)
    expert_ll = if (yl == 0.0)
        log.(p0 + (1 - p0) * exp.(expert_ll_pos))
    else
        log.(0.0 + (1 - p0) * exp.(expert_ll_pos))
    end
    return expert_ll
end
function expert_tn(d::ZIGammaCountExpert, tl::Real, yl::Real, yu::Real, tu::Real)
    expert_tn_pos = LRMoE.expert_tn(LRMoE.GammaCountExpert(d.m, d.s), tl, yl, yu, tu)
    # Deal with zero inflation
    p0 = params(d)[1]
    expert_tn = if (tl == 0.0)
        log.(p0 + (1 - p0) * exp.(expert_tn_pos))
    else
        log.(0.0 + (1 - p0) * exp.(expert_tn_pos))
    end
    return expert_tn
end
function expert_tn_bar(d::ZIGammaCountExpert, tl::Real, yl::Real, yu::Real, tu::Real)
    expert_tn_bar_pos = LRMoE.expert_tn_bar(
        LRMoE.GammaCountExpert(d.m, d.s), tl, yl, yu, tu
    )
    # Deal with zero inflation
    p0 = params(d)[1]
    expert_tn_bar = if (tl > 0.0)
        log.(p0 + (1 - p0) * exp.(expert_tn_bar_pos))
    else
        log.(0.0 + (1 - p0) * exp.(expert_tn_bar_pos))
    end
    return expert_tn_bar
end

function exposurize_expert(d::ZIGammaCountExpert; exposure=1)
    return ZIGammaCountExpert(d.p, d.m, d.s / exposure)
end

## Parameters
params(d::ZIGammaCountExpert) = (d.p, d.m, d.s)
p_zero(d::ZIGammaCountExpert) = d.p
function params_init(y, d::ZIGammaCountExpert)
    p_init = sum(y .== 0.0) / sum(y .>= 0.0)
    pos_idx = (y .> 0.0)

    m_init, s_init = params(params_init(y[pos_idx], GammaCountExpert()))

    try
        ZIGammaCountExpert(p_init, m_init, s_init)
    catch
        ZIGammaCountExpert()
    end
end

## Simululation
function sim_expert(d::ZIGammaCountExpert)
    return (1 .- Distributions.rand(Distributions.Bernoulli(d.p), 1)[1]) .*
           Distributions.rand(LRMoE.GammaCount(d.m, d.s), 1)[1]
end

## penalty
penalty_init(d::ZIGammaCountExpert) = [2.0 10.0 2.0 10.0]
no_penalty_init(d::ZIGammaCountExpert) = [1.0 Inf 1.0 Inf]
function penalize(d::ZIGammaCountExpert, p)
    return (p[1] - 1) * log(d.m) - d.m / p[2] + (p[3] - 1) * log(d.s) - d.s / p[4]
end

## statistics
mean(d::ZIGammaCountExpert) = (1 - d.p) * mean(LRMoE.GammaCount(d.m, d.s))
function var(d::ZIGammaCountExpert)
    return (1 - d.p) * var(LRMoE.GammaCount(d.m, d.s)) +
           d.p * (1 - d.p) * (mean(LRMoE.GammaCount(d.m, d.s)))^2
end
function quantile(d::ZIGammaCountExpert, p)
    return p <= d.p ? 0.0 : quantile(LRMoE.GammaCount(d.m, d.s), p - d.p)
end

## EM: M-Step
function EM_M_expert(d::ZIGammaCountExpert,
    tl, yl, yu, tu,
    exposure,
    z_e_obs, z_e_lat, k_e;
    penalty=true, pen_pararms_jk=[2.0 1.0])

    # Old parameters
    p_old = p_zero(d)

    if p_old > 0.999999
        return d
    end

    expert_ll_pos = fill(0.0, length(yl))
    expert_tn_bar_pos = fill(0.0, length(yl))

    tmp_exp = GammaCountExpert(d.m, d.s)
    for i in 1:length(yl)
        d_expo = exposurize_expert(tmp_exp; exposure=exposure[i])
        expert_ll_pos[i] = expert_ll.(d_expo, tl[i], yl[i], yu[i], tu[i])
        expert_tn_bar_pos[i] = expert_tn_bar.(d_expo, tl[i], yl[i], yu[i], tu[i])
    end

    # Update zero probability
    z_zero_e_obs = z_e_obs .* EM_E_z_zero_obs(yl, p_old, expert_ll_pos)
    z_pos_e_obs = z_e_obs .- z_zero_e_obs
    z_zero_e_lat = z_e_lat .* EM_E_z_zero_lat(tl, p_old, expert_tn_bar_pos)
    z_pos_e_lat = z_e_lat .- z_zero_e_lat
    p_new = EM_M_zero(z_zero_e_obs, z_pos_e_obs, z_zero_e_lat, z_pos_e_lat, k_e)

    # Update parameters: call its positive part
    tmp_update = EM_M_expert(tmp_exp,
        tl, yl, yu, tu,
        exposure,
        z_pos_e_obs, z_pos_e_lat, k_e;
        penalty=penalty, pen_pararms_jk=pen_pararms_jk)

    return ZIGammaCountExpert(p_new, tmp_update.m, tmp_update.s)
end

## EM: M-Step, exact observations
function EM_M_expert_exact(d::ZIGammaCountExpert,
    ye, exposure,
    z_e_obs;
    penalty=true, pen_pararms_jk=[Inf 1.0 Inf])

    # Old parameters
    p_old = p_zero(d)

    if p_old > 0.999999
        return d
    end

    # Update zero probability
    expert_ll_pos = fill(NaN, length(exposure))
    for i in 1:length(exposure)
        expert_ll_pos[i] = expert_ll_exact(
            exposurize_expert(LRMoE.GammaCountExpert(d.m, d.s); exposure=exposure[i]), ye[i]
        )
    end

    z_zero_e_obs = z_e_obs .* EM_E_z_zero_obs(ye, p_old, expert_ll_pos)
    z_pos_e_obs = z_e_obs .- z_zero_e_obs
    p_new = EM_M_zero(z_zero_e_obs, z_pos_e_obs, 0.0, 0.0, 0.0)

    # Update parameters: call its positive part
    tmp_exp = GammaCountExpert(d.m, d.s)
    tmp_update = EM_M_expert_exact(tmp_exp,
        ye, exposure,
        z_pos_e_obs;
        penalty=penalty, pen_pararms_jk=pen_pararms_jk)

    return ZIGammaCountExpert(p_new, tmp_update.m, tmp_update.s)
end