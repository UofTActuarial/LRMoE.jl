struct ZIBurrExpert{T<:Real} <: ZIContinuousExpert
    p::T
    k::T
    c::T
    λ::T
    ZIBurrExpert{T}(p::T, k::T, c::T, λ::T) where {T<:Real} = new{T}(p, k, c, λ)
end


function ZIBurrExpert(p::T, k::T, c::T, λ::T; check_args=true) where {T <: Real}
    check_args && @check_args(ZIBurrExpert, 0 <= p <= 1 && k >= zero(k) && c >= zero(c) && λ > zero(λ))
    return ZIBurrExpert{T}(p, k, c, λ)
end


#### Outer constructors
ZIBurrExpert(p::Real, k::Real, c::Real, λ::Real) = ZIBurrExpert(promote(p, k, c, λ)...)
ZIBurrExpert(p::Integer, k::Integer, c::Integer, λ::Integer) = ZIBurrExpert(float(p), float(k), float(c), float(λ))
ZIBurrExpert() = ZIBurrExpert(0.50, 1.0, 1.0, 1.0)

## Conversion
function convert(::Type{ZIBurrExpert{T}}, p::S, k::S, θ::S) where {T <: Real, S <: Real}
    ZIBurrExpert(T(p), T(k), T(c), T(λ))
end
function convert(::Type{ZIBurrExpert{T}}, d::ZIBurrExpert{S}) where {T <: Real, S <: Real}
    ZIBurrExpert(T(d.p), T(d.k), T(d.c), T(d.λ), check_args=false)
end
copy(d::ZIBurrExpert) = ZIBurrExpert(d.p, d.k, d.c, d.λ, check_args=false)

## Loglikelihood of Expoert
logpdf(d::ZIBurrExpert, x...) = Distributions.logpdf.(LRMoE.Burr(d.k, d.c, d.λ), x...)
pdf(d::ZIBurrExpert, x...) = Distributions.pdf.(LRMoE.Burr(d.k, d.c, d.λ), x...)
logcdf(d::ZIBurrExpert, x...) = Distributions.logcdf.(LRMoE.Burr(d.k, d.c, d.λ), x...)
cdf(d::ZIBurrExpert, x...) = Distributions.cdf.(LRMoE.Burr(d.k, d.c, d.λ), x...)

## expert_ll, etc
expert_ll_exact(d::ZIBurrExpert, x::Real) = (x == 0.) ? log(p_zero(d)) :  log(1-p_zero(d)) + LRMoE.logpdf(d, x)
function expert_ll(d::ZIBurrExpert, tl::Real, yl::Real, yu::Real, tu::Real)
    expert_ll_pos = LRMoE.expert_ll(LRMoE.BurrExpert(d.k, d.c, d.λ), tl, yl, yu, tu)
    # Deal with zero inflation
    p0 = p_zero(d)
    expert_ll = (yl == 0.) ? log.(p0 + (1-p0)*exp.(expert_ll_pos)) : log.(0.0 + (1-p0)*exp.(expert_ll_pos))
    expert_ll = (tu == 0.) ? log.(p0) : expert_ll
    return expert_ll
end
function expert_tn(d::ZIBurrExpert, tl::Real, yl::Real, yu::Real, tu::Real)
    expert_tn_pos = LRMoE.expert_tn(LRMoE.BurrExpert(d.k, d.c, d.λ), tl, yl, yu, tu)
    # Deal with zero inflation
    p0 = p_zero(d)
    expert_tn = (tl == 0.) ? log.(p0 + (1-p0)*exp.(expert_tn_pos)) : log.(0.0 + (1-p0)*exp.(expert_tn_pos))
    expert_tn = (tu == 0.) ? log.(p0) : expert_tn
    return expert_tn
end
function expert_tn_bar(d::ZIBurrExpert, tl::Real, yl::Real, yu::Real, tu::Real)
    expert_tn_bar_pos = LRMoE.expert_tn_bar(LRMoE.BurrExpert(d.k, d.c, d.λ), tl, yl, yu, tu)
    # Deal with zero inflation
    p0 = p_zero(d)
    expert_tn_bar = (tl > 0.) ? log.(p0 + (1-p0)*exp.(expert_tn_bar_pos)) : log.(0.0 + (1-p0)*exp.(expert_tn_bar_pos))
    return expert_tn_bar
end

## Parameters
params(d::ZIBurrExpert) = (d.p, d.k, d.c, d.λ)
p_zero(d::ZIBurrExpert) = d.p
function params_init(y, d::ZIBurrExpert)
    p_init = sum(y .== 0.0) / sum(y .>= 0.0)
    pos_idx = (y .> 0.0)

    c_init, k_init, θ_init = params(params_init(y[pos_idx], BurrExpert()))
    try 
        return ZIBurrExpert(p_init, c_init, k_init, θ_init)
    catch; 
        ZIBurrExpert()
    end
end

## KS stats for parameter initialization
function ks_distance(y, d::ZIBurrExpert)
    p_zero = sum(y .== 0.0) / sum(y .>= 0.0)
    return max(abs(p_zero-d.p), (1-d.p)*HypothesisTests.ksstats(y[y .> 0.0], LRMoE.Burr(d.k, d.c, d.λ))[2])
end

## Simululation
sim_expert(d::ZIBurrExpert, sample_size) = (1 .- Distributions.rand(Distributions.Bernoulli(d.p), sample_size)) .* Distributions.rand(LRMoE.Burr(d.k, d.c, d.λ), sample_size)

## penalty
penalty_init(d::ZIBurrExpert) = [2.0 10.0 2.0 10.0 2.0 10.0]
no_penalty_init(d::ZIBurrExpert) = [1.0 Inf 1.0 Inf 1.0 Inf]
penalize(d::ZIBurrExpert, p) = (p[1]-1)*log(d.k) - d.k/p[2] + (p[3]-1)*log(d.c) - d.c/p[4] + (p[5]-1)*log(d.λ) - d.λ/p[6]

## statistics
mean(d::ZIBurrExpert) = (1-d.p)*mean(LRMoE.Burr(d.k, d.c, d.λ))
var(d::ZIBurrExpert) = (1-d.p)*var(LRMoE.Burr(d.k, d.c, d.λ)) + d.p*(1-d.p)*(mean(LRMoE.Burr(d.k, d.c, d.λ)))^2
quantile(d::ZIBurrExpert, p) = p <= d.p ? 0.0 : quantile(LRMoE.Burr(d.k, d.c, d.λ), p-d.p)
lev(d::ZIBurrExpert, u) = (1-d.p)*lev(BurrExpert(d.k, d.c, d.λ), u)
excess(d::ZIBurrExpert, u) = mean(d) - lev(d, u)

## EM: M-Step
function EM_M_expert(d::ZIBurrExpert,
                     tl, yl, yu, tu,
                     expert_ll_pos,
                     expert_tn_pos,
                     expert_tn_bar_pos,
                     z_e_obs, z_e_lat, k_e;
                     penalty = true, pen_pararms_jk = [1.0 Inf 1.0 Inf])
    
    # Old parameters
    p_old = d.p

    # Update zero probability
    z_zero_e_obs = z_e_obs .* EM_E_z_zero_obs(yl, p_old, expert_ll_pos)
    z_pos_e_obs = z_e_obs .- z_zero_e_obs
    z_zero_e_lat = z_e_lat .* EM_E_z_zero_lat(tl, p_old, expert_tn_bar_pos)
    z_pos_e_lat = z_e_lat .- z_zero_e_lat
    p_new = EM_M_zero(z_zero_e_obs, z_pos_e_obs, z_zero_e_lat, z_pos_e_lat, k_e)

    # Update parameters: call its positive part
    tmp_exp = BurrExpert(d.k, d.c, d.λ)
    tmp_update = EM_M_expert(tmp_exp,
                            tl, yl, yu, tu,
                            expert_ll_pos,
                            expert_tn_pos,
                            expert_tn_bar_pos,
                            # z_e_obs, z_e_lat, k_e,
                            z_pos_e_obs, z_pos_e_lat, k_e,
                            penalty = penalty, pen_pararms_jk = pen_pararms_jk)

    return ZIBurrExpert(p_new, tmp_update.k, tmp_update.c, tmp_update.λ)
end

## EM: M-Step, exact observations
function EM_M_expert_exact(d::ZIBurrExpert,
                     ye,
                     expert_ll_pos,
                     z_e_obs;
                     penalty = true, pen_pararms_jk = [Inf 1.0 Inf])
    
    # Old parameters
    p_old = d.p

    # Update zero probability
    z_zero_e_obs = z_e_obs .* EM_E_z_zero_obs(ye, p_old, expert_ll_pos)
    z_pos_e_obs = z_e_obs .- z_zero_e_obs
    z_zero_e_lat = 0.0
    z_pos_e_lat = 0.0
    p_new = EM_M_zero(z_zero_e_obs, z_pos_e_obs, 0.0, 0.0, 0.0)
        # EM_M_zero(z_zero_e_obs, z_pos_e_obs, z_zero_e_lat, z_pos_e_lat, k_e)

    # Update parameters: call its positive part
    tmp_exp = BurrExpert(d.k, d.c, d.λ)
    tmp_update = EM_M_expert_exact(tmp_exp,
                            ye,
                            expert_ll_pos,
                            z_pos_e_obs;
                            penalty = penalty, pen_pararms_jk = pen_pararms_jk)

    return ZIBurrExpert(p_new, tmp_update.k, tmp_update.c, tmp_update.λ)
end