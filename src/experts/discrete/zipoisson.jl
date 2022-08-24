struct ZIPoissonExpert{T<:Real} <: ZIDiscreteExpert
    p::T
    λ::T
    ZIPoissonExpert{T}(p::T, λ::T) where {T<:Real} = new{T}(p, λ)
end

function ZIPoissonExpert(p::T, λ::T; check_args=true) where {T<:Real}
    check_args && @check_args(ZIPoissonExpert, 0 <= p <= 1 && λ >= zero(λ))
    return ZIPoissonExpert{T}(p, λ)
end

## Outer constructors
ZIPoissonExpert(p::Real, λ::Real) = ZIPoissonExpert(promote(p, λ)...)
ZIPoissonExpert(p::Integer, λ::Integer) = ZIPoissonExpert(float(p), float(λ))
ZIPoissonExpert() = ZIPoissonExpert(0.5, 1.0)

## Conversion
function convert(::Type{ZIPoissonExpert{T}}, p::S, λ::S) where {T<:Real,S<:Real}
    return ZIPoissonExpert(T(p), T(λ))
end
function convert(::Type{ZIPoissonExpert{T}}, d::ZIPoissonExpert{S}) where {T<:Real,S<:Real}
    return ZIPoissonExpert(T(d.p), T(d.λ); check_args=false)
end
copy(d::ZIPoissonExpert) = ZIPoissonExpert(d.p, d.λ; check_args=false)

## Loglikelihood of Expoert
function logpdf(d::ZIPoissonExpert, x...)
    return isinf(x...) ? 0.0 : Distributions.logpdf.(Distributions.Poisson(d.λ), x...)
end
function pdf(d::ZIPoissonExpert, x...)
    return isinf(x...) ? -Inf : Distributions.pdf.(Distributions.Poisson(d.λ), x...)
end
function logcdf(d::ZIPoissonExpert, x...)
    return isinf(x...) ? 0.0 : Distributions.logcdf.(Distributions.Poisson(d.λ), x...)
end
function cdf(d::ZIPoissonExpert, x...)
    return isinf(x...) ? 1.0 : Distributions.cdf.(Distributions.Poisson(d.λ), x...)
end

## expert_ll, etc
function expert_ll_exact(d::ZIPoissonExpert, x::Real)
    return if (x == 0.0)
        log.(p_zero(d) + (1 - p_zero(d)) * exp.(LRMoE.logpdf(LRMoE.PoissonExpert(d.λ), x)))
    else
        log.(1 - p_zero(d)) + LRMoE.logpdf(LRMoE.PoissonExpert(d.λ), x)
    end
end
function expert_ll(d::ZIPoissonExpert, tl::Real, yl::Real, yu::Real, tu::Real)
    expert_ll_pos = LRMoE.expert_ll(LRMoE.PoissonExpert(d.λ), tl, yl, yu, tu)
    # Deal with zero inflation
    p0 = p_zero(d)
    expert_ll = if (yl == 0.0)
        log.(p0 + (1 - p0) * exp.(expert_ll_pos))
    else
        log.(0.0 + (1 - p0) * exp.(expert_ll_pos))
    end
    return expert_ll
end
function expert_tn(d::ZIPoissonExpert, tl::Real, yl::Real, yu::Real, tu::Real)
    expert_tn_pos = LRMoE.expert_tn(LRMoE.PoissonExpert(d.λ), tl, yl, yu, tu)
    # Deal with zero inflation
    p0 = p_zero(d)
    expert_tn = if (tl == 0.0)
        log.(p0 + (1 - p0) * exp.(expert_tn_pos))
    else
        log.(0.0 + (1 - p0) * exp.(expert_tn_pos))
    end
    return expert_tn
end
function expert_tn_bar(d::ZIPoissonExpert, tl::Real, yl::Real, yu::Real, tu::Real)
    expert_tn_bar_pos = LRMoE.expert_tn_bar(LRMoE.PoissonExpert(d.λ), tl, yl, yu, tu)
    # Deal with zero inflation
    p0 = p_zero(d)
    expert_tn_bar = if (tl > 0.0)
        log.(p0 + (1 - p0) * exp.(expert_tn_bar_pos))
    else
        log.(0.0 + (1 - p0) * exp.(expert_tn_bar_pos))
    end
    return expert_tn_bar
end

exposurize_expert(d::ZIPoissonExpert; exposure=1) = ZIPoissonExpert(d.p, d.λ * exposure)

## Parameters
params(d::ZIPoissonExpert) = (d.p, d.λ)
p_zero(d::ZIPoissonExpert) = d.p
function params_init(y, d::ZIPoissonExpert)
    μ, σ2 = mean(y), var(y)
    λp = (σ2 - μ) / μ
    tmp = λp / μ
    p_init = tmp / (1 + tmp)
    λ_init = λp / p_init
    try
        ZIPoissonExpert(p_init, λ_init)
    catch
        ZIPoissonExpert()
    end
end

## Simululation
function sim_expert(d::ZIPoissonExpert)
    return (1 .- Distributions.rand(Distributions.Bernoulli(d.p), 1)[1]) .*
           Distributions.rand(Distributions.Poisson(d.λ), 1)[1]
end

## penalty
penalty_init(d::ZIPoissonExpert) = [2.0 1.0]
no_penalty_init(d::ZIPoissonExpert) = [1.0 Inf]
penalize(d::ZIPoissonExpert, p) = (p[1] - 1) * log(d.λ) - d.λ / p[2]

## statistics
mean(d::ZIPoissonExpert) = (1 - d.p) * mean(Distributions.Poisson(d.λ))
function var(d::ZIPoissonExpert)
    return (1 - d.p) * var(Distributions.Poisson(d.λ)) +
           d.p * (1 - d.p) * (mean(Distributions.Poisson(d.λ)))^2
end
function quantile(d::ZIPoissonExpert, p)
    return p <= d.p ? 0.0 : quantile(Distributions.Poisson(d.λ), p - d.p)
end

## EM: M-Step
function EM_M_expert(d::ZIPoissonExpert,
    tl, yl, yu, tu,
    exposure,
    z_e_obs, z_e_lat, k_e;
    penalty=true, pen_pararms_jk=[2.0 1.0])

    # Old parameters
    λ_old = d.λ
    p_old = p_zero(d)

    if p_old > 0.999999
        return d
    end

    expert_ll_pos = fill(0.0, length(yl))
    expert_tn_bar_pos = fill(0.0, length(yl))

    tmp_exp = PoissonExpert(d.λ)
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

    return ZIPoissonExpert(p_new, tmp_update.λ)
end

## EM: M-Step, exact observations
function EM_M_expert_exact(d::ZIPoissonExpert,
    ye, exposure,
    # expert_ll_pos,
    z_e_obs;
    penalty=true, pen_pararms_jk=[Inf 1.0 Inf])

    # Old parameters
    λ_old = d.λ
    p_old = p_zero(d)

    if p_old > 0.999999
        return d
    end

    # Update zero probability
    expert_ll_pos = fill(NaN, length(exposure))
    for i in 1:length(exposure)
        expert_ll_pos[i] = expert_ll_exact(
            exposurize_expert(LRMoE.PoissonExpert(λ_old); exposure=exposure[i]), ye[i]
        )
    end

    z_zero_e_obs = z_e_obs .* EM_E_z_zero_obs(ye, p_old, expert_ll_pos)
    z_pos_e_obs = z_e_obs .- z_zero_e_obs
    p_new = EM_M_zero(z_zero_e_obs, z_pos_e_obs, 0.0, 0.0, 0.0)
    # EM_M_zero(z_zero_e_obs, z_pos_e_obs, z_zero_e_lat, z_pos_e_lat, k_e)

    # Update parameters: call its positive part
    tmp_exp = PoissonExpert(d.λ)
    tmp_update = EM_M_expert_exact(tmp_exp,
        ye, exposure,
        # expert_ll_pos,
        z_pos_e_obs;
        penalty=penalty, pen_pararms_jk=pen_pararms_jk)

    return ZIPoissonExpert(p_new, tmp_update.λ)
end