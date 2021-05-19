struct ZIGammaCountExpert{T<:Real} <: ZIDiscreteExpert
    p::T
    m::T
    s::T
    ZIGammaCountExpert{T}(p, m, s) where {T<:Real} = new{T}(p, m, s)
end

function ZIGammaCountExpert(p::T, m::T, s::T; check_args=true) where {T <: Real}
    check_args && @check_args(ZIGammaCountExpert, 0 <= p <= 1 && m > zero(m) && s > zero(s))
    return ZIGammaCountExpert{T}(p, m, s)
end

## Outer constructors
ZIGammaCountExpert(p::Real, m::Real, s::Real) = ZIGammaCountExpert(promote(p, m, s)...)
ZIGammaCountExpert(p::Integer, m::Integer, s::Integer) = ZIGammaCountExpert(float(p), float(m), float(s))
ZIGammaCountExpert() = ZIGammaCountExpert(0.50, 2.0, 1.0)

## Conversion
function convert(::Type{ZIGammaCountExpert{T}}, n::S, p::S) where {T <: Real, S <: Real}
    ZIGammaCountExpert(T(p), T(m), T(s))
end
function convert(::Type{ZIGammaCountExpert{T}}, d::ZIGammaCountExpert{S}) where {T <: Real, S <: Real}
    ZIGammaCountExpert(T(d.p), T(d.m), T(d.s), check_args=false)
end
copy(d::ZIGammaCountExpert) = ZIGammaCountExpert(d.p, d.m, d.s, check_args=false)

## Loglikelihood of Expoert
logpdf(d::ZIGammaCountExpert, x...) = isinf(x...) ? 0.0 : Distributions.logpdf.(LRMoE.GammaCount(d.m, d.s), x...)
pdf(d::ZIGammaCountExpert, x...) = isinf(x...) ? -Inf : Distributions.pdf.(LRMoE.GammaCount(d.m, d.s), x...)
logcdf(d::ZIGammaCountExpert, x...) = isinf(x...) ? 0.0 : Distributions.logcdf.(LRMoE.GammaCount(d.m, d.s), x...)
cdf(d::ZIGammaCountExpert, x...) = isinf(x...) ? 1.0 : Distributions.cdf.(LRMoE.GammaCount(d.m, d.s), x...)

## expert_ll, etc
expert_ll_exact(d::ZIGammaCountExpert, x::Real) = (x == 0.) ? log.(p_zero(d) + (1-p_zero(d))*exp.(LRMoE.logpdf(LRMoE.GammaCountExpert(d.m, d.s), x))) : log.(1-p_zero(d)) + LRMoE.logpdf(LRMoE.GammaCountExpert(d.m, d.s), x)
function expert_ll(d::ZIGammaCountExpert, tl::Real, yl::Real, yu::Real, tu::Real)
    expert_ll_pos = LRMoE.expert_ll(LRMoE.GammaCountExpert(d.m, d.s), tl, yl, yu, tu)
    # Deal with zero inflation
    p0 = p_zero(d)
    expert_ll = (yl == 0.) ? log.(p0 + (1-p0)*exp.(expert_ll_pos)) : log.(0.0 + (1-p0)*exp.(expert_ll_pos))
    return expert_ll
end
function expert_tn(d::ZIGammaCountExpert, tl::Real, yl::Real, yu::Real, tu::Real)
    expert_tn_pos = LRMoE.expert_tn(LRMoE.GammaCountExpert(d.m, d.s), tl, yl, yu, tu)
    # Deal with zero inflation
    p0 = params(d)[1]
    expert_tn = (tl == 0.) ? log.(p0 + (1-p0)*exp.(expert_tn_pos)) : log.(0.0 + (1-p0)*exp.(expert_tn_pos))
    return expert_tn
end
function expert_tn_bar(d::ZIGammaCountExpert, tl::Real, yl::Real, yu::Real, tu::Real)
    expert_tn_bar_pos = LRMoE.expert_tn_bar(LRMoE.GammaCountExpert(d.m, d.s), tl, yl, yu, tu)
    # Deal with zero inflation
    p0 = params(d)[1]
    expert_tn_bar = (tl > 0.) ? log.(p0 + (1-p0)*exp.(expert_tn_bar_pos)) : log.(0.0 + (1-p0)*exp.(expert_tn_bar_pos))
    return expert_tn_bar
end

exposurize_expert(d::ZIGammaCountExpert; exposure = 1) = ZIGammaCountExpert(d.p, d.m, d.s/exposure)

## Parameters
params(d::ZIGammaCountExpert) = (d.p, d.m, d.s)
p_zero(d::ZIGammaCountExpert) = d.p
function params_init(y, d::ZIGammaCountExpert)
    
    p_init = sum(y .== 0.0) / sum(y .>= 0.0)
    pos_idx = (y .> 0.0)

    m_init, s_init = params(params_init(y[pos_idx], GammaCountExpert()))

    try 
        ZIGammaCountExpert(p_init, m_init, s_init) 
    catch; 
        ZIGammaCountExpert() 
    end
end

## Simululation
sim_expert(d::ZIGammaCountExpert) = (1 .- Distributions.rand(Distributions.Bernoulli(d.p), 1)[1]) .* Distributions.rand(LRMoE.GammaCount(d.m, d.s), 1)[1]

## penalty
penalty_init(d::ZIGammaCountExpert) = [2.0 10.0 2.0 10.0]
no_penalty_init(d::ZIGammaCountExpert) = [1.0 Inf 1.0 Inf]
penalize(d::ZIGammaCountExpert, p) = (p[1]-1)*log(d.m) - d.m/p[2] + (p[3]-1)*log(d.s) - d.s/p[4]

## statistics
mean(d::ZIGammaCountExpert) = (1-d.p)*mean(LRMoE.GammaCount(d.m, d.s))
var(d::ZIGammaCountExpert) = (1-d.p)*var(LRMoE.GammaCount(d.m, d.s)) + d.p*(1-d.p)*(mean(LRMoE.GammaCount(d.m, d.s)))^2
quantile(d::ZIGammaCountExpert, p) = p <= d.p ? 0.0 : quantile(LRMoE.GammaCount(d.m, d.s), p-d.p)

## EM: M-Step
function EM_M_expert(d::ZIGammaCountExpert,
                    tl, yl, yu, tu,
                    expert_ll_pos,
                    expert_tn_pos,
                    expert_tn_bar_pos,
                    z_e_obs, z_e_lat, k_e;
                    penalty = true, pen_pararms_jk = [2.0 1.0])

    # Old parameters
    p_old = d.p

    # Update zero probability
    z_zero_e_obs = z_e_obs .* EM_E_z_zero_obs(yl, p_old, expert_ll_pos)
    z_pos_e_obs = z_e_obs .- z_zero_e_obs
    z_zero_e_lat = z_e_lat .* EM_E_z_zero_lat(tl, p_old, expert_tn_bar_pos)
    z_pos_e_lat = z_e_lat .- z_zero_e_lat
    p_new = EM_M_zero(z_zero_e_obs, z_pos_e_obs, z_zero_e_lat, z_pos_e_lat, k_e)

    # Update parameters: call its positive part
    tmp_exp = GammaCountExpert(d.m, d.s)
    tmp_update = EM_M_expert(tmp_exp,
                            tl, yl, yu, tu,
                            expert_ll_pos,
                            expert_tn_pos,
                            expert_tn_bar_pos,
                            # z_e_obs, z_e_lat, k_e,
                            z_pos_e_obs, z_pos_e_lat, k_e,
                            penalty = penalty, pen_pararms_jk = pen_pararms_jk)

    return ZIGammaCountExpert(p_new, tmp_update.m, tmp_update.s)

end

## EM: M-Step, exact observations
function EM_M_expert_exact(d::ZIGammaCountExpert,
                    ye, exposure,
                    z_e_obs; 
                    penalty = true, pen_pararms_jk = [Inf 1.0 Inf])

    # Old parameters
    p_old = p_zero(d)

    # Update zero probability
    expert_ll_pos = fill(NaN, length(exposure))
    for i in 1:length(exposure)
        expert_ll_pos[i] = expert_ll_exact(exposurize_expert(LRMoE.GammaCountExpert(d.m, d.s), exposure = exposure[i]), ye[i])
    end

    z_zero_e_obs = z_e_obs .* EM_E_z_zero_obs(ye, p_old, expert_ll_pos)
    z_pos_e_obs = z_e_obs .- z_zero_e_obs
    p_new = EM_M_zero(z_zero_e_obs, z_pos_e_obs, 0.0, 0.0, 0.0)

    # Update parameters: call its positive part
    tmp_exp = GammaCountExpert(d.m, d.s)
    tmp_update = EM_M_expert_exact(tmp_exp,
                            ye, exposure,
                            z_pos_e_obs,
                            penalty = penalty, pen_pararms_jk = pen_pararms_jk)

    return ZIGammaCountExpert(p_new, tmp_update.m, tmp_update.s)

end