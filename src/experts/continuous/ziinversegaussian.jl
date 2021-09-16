struct ZIInverseGaussianExpert{T<:Real} <: ZIContinuousExpert
    p::T
    μ::T
    λ::T
    ZIInverseGaussianExpert{T}(p::T, µ::T, λ::T) where {T<:Real} = new{T}(p, µ, λ)
end


function ZIInverseGaussianExpert(p::T, μ::T, λ::T; check_args=true) where {T <: Real}
    check_args && @check_args(ZIInverseGaussianExpert, 0 <= p <= 1 && μ >= zero(μ) && λ > zero(λ))
    return ZIInverseGaussianExpert{T}(p, μ, λ)
end


#### Outer constructors
ZIInverseGaussianExpert(p::Real, μ::Real, λ::Real) = ZIInverseGaussianExpert(promote(p, μ, λ)...)
ZIInverseGaussianExpert(p::Integer, μ::Integer, λ::Integer) = ZIInverseGaussianExpert(float(p), float(μ), float(λ))
ZIInverseGaussianExpert() = ZIInverseGaussianExpert(0.5, 1.0, 1.0)

## Conversion
function convert(::Type{ZIInverseGaussianExpert{T}}, p::S, μ::S, λ::S) where {T <: Real, S <: Real}
    λ(T(p), T(μ), T(λ))
end
function convert(::Type{ZIInverseGaussianExpert{T}}, d::ZIInverseGaussianExpert{S}) where {T <: Real, S <: Real}
    ZIInverseGaussianExpert(T(d.p), T(d.μ), T(d.λ), check_args=false)
end
copy(d::ZIInverseGaussianExpert) = ZIInverseGaussianExpert(d.p, d.μ, d.λ, check_args=false)

## Loglikelihood of Expoert
logpdf(d::ZIInverseGaussianExpert, x...) = isinf(x...) ? -Inf : Distributions.logpdf.(Distributions.InverseGaussian(d.μ, d.λ), x...)
pdf(d::ZIInverseGaussianExpert, x...) = isinf(x...) ? 0.0 : Distributions.pdf.(Distributions.InverseGaussian(d.μ, d.λ), x...)
logcdf(d::ZIInverseGaussianExpert, x...) = isinf(x...) ? 0.0 : Distributions.logcdf.(Distributions.InverseGaussian(d.μ, d.λ), x...)
cdf(d::ZIInverseGaussianExpert, x...) = isinf(x...) ? 1.0 : Distributions.cdf.(Distributions.InverseGaussian(d.μ, d.λ), x...)

expert_ll_exact(d::ZIInverseGaussianExpert, x::Real) = (x == 0.) ? log(p_zero(d)) :  log(1-p_zero(d)) + LRMoE.logpdf(d, x)
function expert_ll(d::ZIInverseGaussianExpert, tl::Real, yl::Real, yu::Real, tu::Real)
    expert_ll_pos = LRMoE.expert_ll(LRMoE.InverseGaussianExpert(d.μ, d.λ), tl, yl, yu, tu)
    # Deal with zero inflation
    p0 = p_zero(d)
    expert_ll = (yl == 0.) ? log.(p0 + (1-p0)*exp.(expert_ll_pos)) : log.(0.0 + (1-p0)*exp.(expert_ll_pos))
    expert_ll = (tu == 0.) ? log.(p0) : expert_ll
    return expert_ll
end
function expert_tn(d::ZIInverseGaussianExpert, tl::Real, yl::Real, yu::Real, tu::Real)
    expert_tn_pos = LRMoE.expert_tn(LRMoE.InverseGaussianExpert(d.μ, d.λ), tl, yl, yu, tu)
    # Deal with zero inflation
    p0 = p_zero(d)
    expert_tn = (tl == 0.) ? log.(p0 + (1-p0)*exp.(expert_tn_pos)) : log.(0.0 + (1-p0)*exp.(expert_tn_pos))
    expert_tn = (tu == 0.) ? log.(p0) : expert_tn
    return expert_tn
end
function expert_tn_bar(d::ZIInverseGaussianExpert, tl::Real, yl::Real, yu::Real, tu::Real)
    expert_tn_bar_pos = LRMoE.expert_tn_bar(LRMoE.InverseGaussianExpert(d.μ, d.λ), tl, yl, yu, tu)
    # Deal with zero inflation
    p0 = p_zero(d)
    expert_tn_bar = (tl > 0.) ? log.(p0 + (1-p0)*exp.(expert_tn_bar_pos)) : log.(0.0 + (1-p0)*exp.(expert_tn_bar_pos))
    return expert_tn_bar
end

exposurize_expert(d::ZIInverseGaussianExpert; exposure = 1) = d

## Parameters
params(d::ZIInverseGaussianExpert) = (d.p, d.μ, d.λ)
p_zero(d::ZIInverseGaussianExpert) = d.p
function params_init(y, d::ZIInverseGaussianExpert)
    p_init = sum(y .== 0.0) / sum(y .>= 0.0)
    pos_idx = (y .> 0.0)
    μ, σ2 = mean(y[pos_idx]), var(y[pos_idx])
    μ_init = μ
    λ_init = μ^3 / σ2
    if isnan(μ_init) || isnan(λ_init)
        return ZIInverseGaussianExpert()
    else
        return ZIInverseGaussianExpert(p_init, μ_init, λ_init)
    end
end

## KS stats for parameter initialization
function ks_distance(y, d::ZIInverseGaussianExpert)
    p_zero = sum(y .== 0.0) / sum(y .>= 0.0)
    return max(abs(p_zero-d.p), (1-d.p)*HypothesisTests.ksstats(y[y .> 0.0], Distributions.InverseGaussian(d.μ, d.λ))[2])
end

## Simululation
sim_expert(d::ZIInverseGaussianExpert) = (1 .- Distributions.rand(Distributions.Bernoulli(d.p), 1)[1]) .* Distributions.rand(Distributions.InverseGaussian(d.μ, d.λ), 1)[1]

## penalty
penalty_init(d::ZIInverseGaussianExpert) = [1.0 Inf 1.0 Inf]
no_penalty_init(d::ZIInverseGaussianExpert) = [1.0 Inf 1.0 Inf]
penalize(d::ZIInverseGaussianExpert, p) = (p[1]-1)*log(d.μ) - d.μ/p[2] + (p[3]-1)*log(d.λ) - d.λ/p[4]

## statistics
mean(d::ZIInverseGaussianExpert) = (1-d.p)*mean(Distributions.InverseGaussian(d.μ, d.λ))
var(d::ZIInverseGaussianExpert) = (1-d.p)*var(Distributions.InverseGaussian(d.μ, d.λ)) + d.p*(1-d.p)*(mean(Distributions.InverseGaussian(d.μ, d.λ)))^2
quantile(d::ZIInverseGaussianExpert, p) = p <= d.p ? 0.0 : quantile(Distributions.InverseGaussian(d.μ, d.λ), p-d.p)
lev(d::ZIInverseGaussianExpert, u) = (1-d.p)*lev(InverseGaussianExpert(d.μ, d.λ), u)
excess(d::ZIInverseGaussianExpert, u) = mean(d) - lev(d, u)

## EM: M-Step
function EM_M_expert(d::ZIInverseGaussianExpert,
                     tl, yl, yu, tu,
                     exposure,
                     z_e_obs, z_e_lat, k_e;
                     penalty = true, pen_pararms_jk = [Inf 1.0 Inf])
    
    # Old parameters
    p_old = p_zero(d)

    if p_old > 0.999999
        return d
    end

    # Update zero probability
    expert_ll_pos = expert_ll.(LRMoE.InverseGaussianExpert(d.μ, d.λ), tl, yl, yu, tu)
    expert_tn_bar_pos = expert_tn_bar.(LRMoE.InverseGaussianExpert(d.μ, d.λ), tl, yl, yu, tu)

    z_zero_e_obs = z_e_obs .* EM_E_z_zero_obs(yl, p_old, expert_ll_pos)
    z_pos_e_obs = z_e_obs .- z_zero_e_obs
    z_zero_e_lat = z_e_lat .* EM_E_z_zero_lat(tl, p_old, expert_tn_bar_pos)
    z_pos_e_lat = z_e_lat .- z_zero_e_lat
    p_new = EM_M_zero(z_zero_e_obs, z_pos_e_obs, z_zero_e_lat, z_pos_e_lat, k_e)

    # Update parameters: call its positive part
    tmp_exp = InverseGaussianExpert(d.μ, d.λ)
    tmp_update = EM_M_expert(tmp_exp,
                            tl, yl, yu, tu,
                            exposure,
                            z_pos_e_obs, z_pos_e_lat, k_e,
                            penalty = penalty, pen_pararms_jk = pen_pararms_jk)

    return ZIInverseGaussianExpert(p_new, tmp_update.μ, tmp_update.λ)
end

## EM: M-Step, exact observations
function EM_M_expert_exact(d::ZIInverseGaussianExpert,
                     ye, exposure,
                     z_e_obs;
                     penalty = true, pen_pararms_jk = [Inf 1.0 Inf])
    
    # Old parameters
    p_old = p_zero(d)

    if p_old > 0.999999
        return d
    end

    # Update zero probability
    expert_ll_pos = expert_ll_exact.(LRMoE.InverseGaussianExpert(d.μ, d.λ), ye)

    z_zero_e_obs = z_e_obs .* EM_E_z_zero_obs(ye, p_old, expert_ll_pos)
    z_pos_e_obs = z_e_obs .- z_zero_e_obs
    p_new = EM_M_zero(z_zero_e_obs, z_pos_e_obs, 0.0, 0.0, 0.0)

    # Update parameters: call its positive part
    tmp_exp = InverseGaussianExpert(d.μ, d.λ)
    tmp_update = EM_M_expert_exact(tmp_exp,
                            ye, exposure,
                            z_pos_e_obs;
                            penalty = penalty, pen_pararms_jk = pen_pararms_jk)

    return ZIInverseGaussianExpert(p_new, tmp_update.μ, tmp_update.λ)
end