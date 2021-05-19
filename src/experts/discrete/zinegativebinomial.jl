struct ZINegativeBinomialExpert{T<:Real} <: ZIDiscreteExpert
    p0::T
    n::T
    p::T
    ZINegativeBinomialExpert{T}(p0, n, p) where {T<:Real} = new{T}(p0, n, p)
end

function ZINegativeBinomialExpert(p0::T, n::T, p::T; check_args=true) where {T <: Real}
    check_args && @check_args(ZINegativeBinomialExpert, 0 <= p0 <= 1 && 0 <= p <= 1 && n > zero(n))
    return ZINegativeBinomialExpert{T}(p0, n, p)
end

## Outer constructors
ZINegativeBinomialExpert(p0::Real, n::Real, p::Real) = ZINegativeBinomialExpert(promote(p0, n, p)...)
ZINegativeBinomialExpert(p0::Integer, n::Integer, p::Integer) = ZINegativeBinomialExpert(float(p0), float(n), float(p))
ZINegativeBinomialExpert() = ZINegativeBinomialExpert(0.50, 1, 0.50)

## Conversion
function convert(::Type{ZINegativeBinomialExpert{T}}, p0::S, n::S, p::S) where {T <: Real, S <: Real}
    ZINegativeBinomialExpert(T(p0), T(n), T(p))
end
function convert(::Type{ZINegativeBinomialExpert{T}}, d::ZINegativeBinomialExpert{S}) where {T <: Real, S <: Real}
    ZINegativeBinomialExpert(T(d.p0), T(d.n), T(d.p), check_args=false)
end
copy(d::ZINegativeBinomialExpert) = ZINegativeBinomialExpert(d.p0, d.n, d.p, check_args=false)

## Loglikelihood of Expoert
logpdf(d::ZINegativeBinomialExpert, x...) = isinf(x...) ? -Inf : Distributions.logpdf.(Distributions.NegativeBinomial(d.n, d.p), x...)
pdf(d::ZINegativeBinomialExpert, x...) = isinf(x...) ? 0.0 : Distributions.pdf.(Distributions.NegativeBinomial(d.n, d.p), x...)
logcdf(d::ZINegativeBinomialExpert, x...) = isinf(x...) ? 0.0 : Distributions.logcdf.(Distributions.NegativeBinomial(d.n, d.p), x...)
cdf(d::ZINegativeBinomialExpert, x...) = isinf(x...) ? 1.0 : Distributions.cdf.(Distributions.NegativeBinomial(d.n, d.p), x...)

## expert_ll, etc
expert_ll_exact(d::ZINegativeBinomialExpert, x::Real) = (x == 0.) ? log.(p_zero(d) + (1-p_zero(d))*exp.(LRMoE.logpdf(LRMoE.NegativeBinomialExpert(d.n, d.p), x))) : log.(1-p_zero(d)) + LRMoE.logpdf(LRMoE.NegativeBinomialExpert(d.n, d.p), x)
function expert_ll(d::ZINegativeBinomialExpert, tl::Real, yl::Real, yu::Real, tu::Real)
    expert_ll_pos = LRMoE.expert_ll(LRMoE.NegativeBinomialExpert(d.n, d.p), tl, yl, yu, tu)
    # Deal with zero inflation
    p0 = p_zero(d)
    expert_ll = (yl == 0.) ? log.(p0 + (1-p0)*exp.(expert_ll_pos)) : log.(0.0 + (1-p0)*exp.(expert_ll_pos))
    return expert_ll
end
function expert_tn(d::ZINegativeBinomialExpert, tl::Real, yl::Real, yu::Real, tu::Real)
    expert_tn_pos = LRMoE.expert_tn(LRMoE.NegativeBinomialExpert(d.n, d.p), tl, yl, yu, tu)
    # Deal with zero inflation
    p0 = params(d)[1]
    expert_tn = (tl == 0.) ? log.(p0 + (1-p0)*exp.(expert_tn_pos)) : log.(0.0 + (1-p0)*exp.(expert_tn_pos))
    return expert_tn
end
function expert_tn_bar(d::ZINegativeBinomialExpert, tl::Real, yl::Real, yu::Real, tu::Real)
    expert_tn_bar_pos = LRMoE.expert_tn_bar(LRMoE.NegativeBinomialExpert(d.n, d.p), tl, yl, yu, tu)
    # Deal with zero inflation
    p0 = params(d)[1]
    expert_tn_bar = (tl > 0.) ? log.(p0 + (1-p0)*exp.(expert_tn_bar_pos)) : log.(0.0 + (1-p0)*exp.(expert_tn_bar_pos))
    return expert_tn_bar
end

exposurize_expert(d::ZINegativeBinomialExpert; exposure = 1) = ZINegativeBinomialExpert(d.p0, d.n*exposure, d.p)

## Parameters
params(d::ZINegativeBinomialExpert) = (d.p0, d.n, d.p)
p_zero(d::ZINegativeBinomialExpert) = d.p0
function params_init(y, d::ZINegativeBinomialExpert)
    pos_idx = (y .> 0.0)
    μ, σ2 = mean(y[pos_idx]), var(y[pos_idx])
    p_init = μ / σ2
    n_init = μ*p_init/(1-p_init)
    p0_init = 1 - mean(y)/μ
    try 
        ZINegativeBinomialExpert(p0_init, n_init, p_init) 
    catch; 
        ZINegativeBinomialExpert() 
    end
end

## Simululation
sim_expert(d::ZINegativeBinomialExpert) = (1 .- Distributions.rand(Distributions.Bernoulli(d.p0), 1)[1]) .* Distributions.rand(Distributions.NegativeBinomial(d.n, d.p), 1)[1]

## penalty
penalty_init(d::ZINegativeBinomialExpert) = [2.0 10.0]
no_penalty_init(d::ZINegativeBinomialExpert) = [1.0 Inf]
penalize(d::ZINegativeBinomialExpert, p) = (p[1]-1)*log(d.n) - d.n/p[2]

## statistics
mean(d::ZINegativeBinomialExpert) = (1-d.p0)*mean(Distributions.NegativeBinomial(d.n, d.p))
var(d::ZINegativeBinomialExpert) = (1-d.p0)*var(Distributions.NegativeBinomial(d.n, d.p)) + d.p0*(1-d.p0)*(mean(Distributions.NegativeBinomial(d.n, d.p)))^2
quantile(d::ZINegativeBinomialExpert, p) = p <= d.p0 ? 0.0 :  quantile(Distributions.NegativeBinomial(d.n, d.p), p-d.p0)

## EM: M-Step
function EM_M_expert(d::ZINegativeBinomialExpert,
                    tl, yl, yu, tu,
                    expert_ll_pos,
                    expert_tn_pos,
                    expert_tn_bar_pos,
                    z_e_obs, z_e_lat, k_e;
                    penalty = true, pen_pararms_jk = [2.0 1.0])

    # Old parameters
    p_old = d.p0

    # Update zero probability
    z_zero_e_obs = z_e_obs .* EM_E_z_zero_obs(yl, p_old, expert_ll_pos)
    z_pos_e_obs = z_e_obs .- z_zero_e_obs
    z_zero_e_lat = z_e_lat .* EM_E_z_zero_lat(tl, p_old, expert_tn_bar_pos)
    z_pos_e_lat = z_e_lat .- z_zero_e_lat
    p_new = EM_M_zero(z_zero_e_obs, z_pos_e_obs, z_zero_e_lat, z_pos_e_lat, k_e)

    # Update parameters: call its positive part
    tmp_exp = NegativeBinomialExpert(d.n, d.p)
    tmp_update = EM_M_expert(tmp_exp,
                            tl, yl, yu, tu,
                            expert_ll_pos,
                            expert_tn_pos,
                            expert_tn_bar_pos,
                            # z_e_obs, z_e_lat, k_e,
                            z_pos_e_obs, z_pos_e_lat, k_e,
                            penalty = penalty, pen_pararms_jk = pen_pararms_jk)

    return ZINegativeBinomialExpert(p_new, tmp_update.n, tmp_update.p)

end

## EM: M-Step, exact observations
function EM_M_expert_exact(d::ZINegativeBinomialExpert,
                    ye, exposure,
                    z_e_obs; 
                    penalty = true, pen_pararms_jk = [2.0 1.0])

    # Old parameters
    p_old = p_zero(d)

    # Update zero probability
    expert_ll_pos = fill(NaN, length(exposure))
    for i in 1:length(exposure)
        expert_ll_pos[i] = expert_ll_exact(exposurize_expert(LRMoE.NegativeBinomialExpert(d.n, d.p), exposure = exposure[i]), ye[i])
    end

    z_zero_e_obs = z_e_obs .* EM_E_z_zero_obs(ye, p_old, expert_ll_pos)
    z_pos_e_obs = z_e_obs .- z_zero_e_obs
    p_new = EM_M_zero(z_zero_e_obs, z_pos_e_obs, 0.0, 0.0, 0.0)

    # Update parameters: call its positive part
    tmp_exp = NegativeBinomialExpert(d.n, d.p)
    tmp_update = EM_M_expert_exact(tmp_exp,
                            ye, exposure,
                            z_pos_e_obs,
                            penalty = penalty, pen_pararms_jk = pen_pararms_jk)

    return ZINegativeBinomialExpert(p_new, tmp_update.n, tmp_update.p)

end