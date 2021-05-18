struct ZILogNormalExpert{T<:Real} <: ZIContinuousExpert
    p::T
    μ::T
    σ::T
    ZILogNormalExpert{T}(p::T, µ::T, σ::T) where {T<:Real} = new{T}(p, µ, σ)
end


function ZILogNormalExpert(p::T, μ::T, σ::T; check_args=true) where {T <: Real}
    check_args && @check_args(ZILogNormalExpert, 0 <= p <= 1 && σ >= zero(σ))
    return ZILogNormalExpert{T}(p, μ, σ)
end


#### Outer constructors
ZILogNormalExpert(p::Real, μ::Real, σ::Real) = ZILogNormalExpert(promote(p, μ, σ)...)
ZILogNormalExpert(p::Integer, μ::Integer, σ::Integer) = ZILogNormalExpert(float(p), float(μ), float(σ))
ZILogNormalExpert() = ZILogNormalExpert(0.5, 0.0, 1.0)

## Conversion
function convert(::Type{ZILogNormalExpert{T}}, p::S, μ::S, σ::S) where {T <: Real, S <: Real}
    ZILogNormalExpert(T(p), T(μ), T(σ))
end
function convert(::Type{ZILogNormalExpert{T}}, d::ZILogNormalExpert{S}) where {T <: Real, S <: Real}
    ZILogNormalExpert(T(d.p), T(d.μ), T(d.σ), check_args=false)
end
copy(d::ZILogNormalExpert) = ZILogNormalExpert(d.p, d.μ, d.σ, check_args=false)

## Loglikelihood of Expoert
logpdf(d::ZILogNormalExpert, x...) = Distributions.logpdf.(Distributions.LogNormal(d.μ, d.σ), x...)
pdf(d::ZILogNormalExpert, x...) = Distributions.pdf.(Distributions.LogNormal(d.μ, d.σ), x...)
logcdf(d::ZILogNormalExpert, x...) = Distributions.logcdf.(Distributions.LogNormal(d.μ, d.σ), x...)
cdf(d::ZILogNormalExpert, x...) = Distributions.cdf.(Distributions.LogNormal(d.μ, d.σ), x...)

expert_ll_exact(d::ZILogNormalExpert, x::Real) = (x == 0.) ? log(p_zero(d)) :  log(1-p_zero(d)) + LRMoE.logpdf(d, x)
function expert_ll(d::ZILogNormalExpert, tl::Real, yl::Real, yu::Real, tu::Real)
    expert_ll_pos = LRMoE.expert_ll(LRMoE.LogNormalExpert(d.μ, d.σ), tl, yl, yu, tu)
    # Deal with zero inflation
    p0 = p_zero(d)
    expert_ll = (yl == 0.) ? log.(p0 + (1-p0)*exp.(expert_ll_pos)) : log.(0.0 + (1-p0)*exp.(expert_ll_pos))
    expert_ll = (tu == 0.) ? log.(p0) : expert_ll
    return expert_ll
end
function expert_tn(d::ZILogNormalExpert, tl::Real, yl::Real, yu::Real, tu::Real)
    expert_tn_pos = LRMoE.expert_tn(LRMoE.LogNormalExpert(d.μ, d.σ), tl, yl, yu, tu)
    # Deal with zero inflation
    p0 = p_zero(d)
    expert_tn = (tl == 0.) ? log.(p0 + (1-p0)*exp.(expert_tn_pos)) : log.(0.0 + (1-p0)*exp.(expert_tn_pos))
    expert_tn = (tu == 0.) ? log.(p0) : expert_tn
    return expert_tn
end
function expert_tn_bar(d::ZILogNormalExpert, tl::Real, yl::Real, yu::Real, tu::Real)
    expert_tn_bar_pos = LRMoE.expert_tn_bar(LRMoE.LogNormalExpert(d.μ, d.σ), tl, yl, yu, tu)
    # Deal with zero inflation
    p0 = p_zero(d)
    expert_tn_bar = (tl > 0.) ? log.(p0 + (1-p0)*exp.(expert_tn_bar_pos)) : log.(0.0 + (1-p0)*exp.(expert_tn_bar_pos))
    return expert_tn_bar
end

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
    return max(abs(p_zero-d.p), (1-d.p)*HypothesisTests.ksstats(y[y .> 0.0], Distributions.LogNormal(d.μ, d.σ))[2])
end

## Simululation
sim_expert(d::ZILogNormalExpert, sample_size) = (1 .- Distributions.rand(Distributions.Bernoulli(d.p), sample_size)) .* Distributions.rand(Distributions.LogNormal(d.μ, d.σ), sample_size)

## penalty
penalty_init(d::ZILogNormalExpert) = [Inf 1.0 Inf]
no_penalty_init(d::ZILogNormalExpert) = [Inf 1.0 Inf]
penalize(d::ZILogNormalExpert, p) = (d.μ/p[1])^2 + (p[2]-1)*log(d.σ) - d.σ/p[3]

## statistics
mean(d::ZILogNormalExpert) = (1-d.p)*mean(Distributions.LogNormal(d.μ, d.σ))
var(d::ZILogNormalExpert) = (1-d.p)*var(Distributions.LogNormal(d.μ, d.σ)) + d.p*(1-d.p)*(mean(Distributions.LogNormal(d.μ, d.σ)))^2
quantile(d::ZILogNormalExpert, p) = p <= d.p ? 0.0 : quantile(Distributions.LogNormal(d.μ, d.σ), p-d.p)
lev(d::ZILogNormalExpert, u) = (1-d.p) * lev(LogNormalExpert(d.μ, d.σ), u)
excess(d::ZILogNormalExpert, u) = mean(d) - lev(d, u)

## EM: M-Step
function EM_M_expert(d::ZILogNormalExpert,
                     tl, yl, yu, tu,
                     expert_ll_pos,
                     expert_tn_pos,
                     expert_tn_bar_pos,
                     z_e_obs, z_e_lat, k_e;
                     penalty = true, pen_pararms_jk = [Inf 1.0 Inf])
    
    # Old parameters
    μ_old = d.μ
    σ_old = d.σ
    p_old = d.p

    # Update zero probability
    z_zero_e_obs = z_e_obs .* EM_E_z_zero_obs(yl, p_old, expert_ll_pos)
    z_pos_e_obs = z_e_obs .- z_zero_e_obs
    z_zero_e_lat = z_e_lat .* EM_E_z_zero_lat(tl, p_old, expert_tn_bar_pos)
    z_pos_e_lat = z_e_lat .- z_zero_e_lat
    p_new = EM_M_zero(z_zero_e_obs, z_pos_e_obs, z_zero_e_lat, z_pos_e_lat, k_e)

    # Update parameters: call its positive part
    tmp_exp = LogNormalExpert(d.μ, d.σ)
    tmp_update = EM_M_expert(tmp_exp,
                            tl, yl, yu, tu,
                            expert_ll_pos,
                            expert_tn_pos,
                            expert_tn_bar_pos,
                            # z_e_obs, z_e_lat, k_e,
                            z_pos_e_obs, z_pos_e_lat, k_e,
                            penalty = penalty, pen_pararms_jk = pen_pararms_jk)

    return ZILogNormalExpert(p_new, tmp_update.μ, tmp_update.σ)
end

## EM: M-Step, exact observations
function EM_M_expert_exact(d::ZILogNormalExpert,
                     ye,
                     expert_ll_pos,
                     z_e_obs;
                     penalty = true, pen_pararms_jk = [Inf 1.0 Inf])
    
    # Old parameters
    μ_old = d.μ
    σ_old = d.σ
    p_old = d.p

    # Update zero probability
    z_zero_e_obs = z_e_obs .* EM_E_z_zero_obs(ye, p_old, expert_ll_pos)
    z_pos_e_obs = z_e_obs .- z_zero_e_obs
    z_zero_e_lat = 0.0
    z_pos_e_lat = 0.0
    p_new = EM_M_zero(z_zero_e_obs, z_pos_e_obs, 0.0, 0.0, 0.0)
        # EM_M_zero(z_zero_e_obs, z_pos_e_obs, z_zero_e_lat, z_pos_e_lat, k_e)

    # Update parameters: call its positive part
    tmp_exp = LogNormalExpert(d.μ, d.σ)
    tmp_update = EM_M_expert_exact(tmp_exp,
                            ye,
                            expert_ll_pos,
                            z_pos_e_obs;
                            penalty = penalty, pen_pararms_jk = pen_pararms_jk)

    return ZILogNormalExpert(p_new, tmp_update.μ, tmp_update.σ)
end