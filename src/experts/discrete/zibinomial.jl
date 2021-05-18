struct ZIBinomialExpert{T<:Real} <: NonZIDiscreteExpert
    p0::T
    n::Int
    p::T
    ZIBinomialExpert{T}(p0, n, p) where {T<:Real} = new{T}(p0, n, p)
end

function ZIBinomialExpert(p0::T, n::Integer, p::T; check_args=true) where {T <: Real}
    check_args && @check_args(ZIBinomialExpert, 0 <= p0 <= 1 && 0 <= p <= 1 && isa(n, Integer))
    return ZIBinomialExpert{T}(p0, n, p)
end

## Outer constructors
# ZIBinomialExpert(p0::Real, n::Integer, p::Real) = ZIBinomialExpert(p0, n, float(p))
ZIBinomialExpert(p0::Integer, n::Integer, p::Integer) = ZIBinomialExpert(float(p0), n, float(p))
ZIBinomialExpert() = ZIBinomialExpert(0.50, 2, 0.50)

## Conversion
function convert(::Type{ZIBinomialExpert{T}}, p0::S, n::Int, p::S) where {T <: Real, S <: Real}
    ZIBinomialExpert(T(p0), n, T(p))
end
function convert(::Type{ZIBinomialExpert{T}}, d::ZIBinomialExpert{S}) where {T <: Real, S <: Real}
    ZIBinomialExpert(T(d.p0), d.n, T(d.p), check_args=false)
end
copy(d::ZIBinomialExpert) = ZIBinomialExpert(d.p0, d.n, d.p, check_args=false)

## Loglikelihood of Expoert
logpdf(d::ZIBinomialExpert, x...) = isinf(x...) ? 0.0 : Distributions.logpdf.(Distributions.Binomial(d.n, d.p), x...)
pdf(d::ZIBinomialExpert, x...) = isinf(x...) ? -Inf : Distributions.pdf.(Distributions.Binomial(d.n, d.p), x...)
logcdf(d::ZIBinomialExpert, x...) = isinf(x...) ? 0.0 : Distributions.logcdf.(Distributions.Binomial(d.n, d.p), x...)
cdf(d::ZIBinomialExpert, x...) = isinf(x...) ? 1.0 : Distributions.cdf.(Distributions.Binomial(d.n, d.p), x...)

## expert_ll, etc
expert_ll_exact(d::ZIBinomialExpert, x::Real) = (x == 0.) ? log.(p_zero(d) + (1-p_zero(d))*exp.(LRMoE.logpdf(d, x))) : log.(1-p_zero(d)) + LRMoE.logpdf(d, x)
function expert_ll(d::ZIBinomialExpert, tl::Real, yl::Real, yu::Real, tu::Real)
    expert_ll_pos = LRMoE.expert_ll(LRMoE.BinomialExpert(d.n, d.p), tl, yl, yu, tu)
    # Deal with zero inflation
    p0 = p_zero(d)
    expert_ll = (yl == 0.) ? log.(p0 + (1-p0)*exp.(expert_ll_pos)) : log.(0.0 + (1-p0)*exp.(expert_ll_pos))
    return expert_ll
end
function expert_tn(d::ZIBinomialExpert, tl::Real, yl::Real, yu::Real, tu::Real)
    expert_tn_pos = LRMoE.expert_tn(LRMoE.BinomialExpert(d.n, d.p), tl, yl, yu, tu)
    # Deal with zero inflation
    p0 = params(d)[1]
    expert_tn = (tl == 0.) ? log.(p0 + (1-p0)*exp.(expert_tn_pos)) : log.(0.0 + (1-p0)*exp.(expert_tn_pos))
    return expert_tn
end
function expert_tn_bar(d::ZIBinomialExpert, tl::Real, yl::Real, yu::Real, tu::Real)
    expert_tn_bar_pos = LRMoE.expert_tn_bar(LRMoE.BinomialExpert(d.n, d.p), tl, yl, yu, tu)
    # Deal with zero inflation
    p0 = params(d)[1]
    expert_tn_bar = (tl > 0.) ? log.(p0 + (1-p0)*exp.(expert_tn_bar_pos)) : log.(0.0 + (1-p0)*exp.(expert_tn_bar_pos))
    return expert_tn_bar
end

## Parameters
params(d::ZIBinomialExpert) = (d.p0, d.n, d.p)
p_zero(d::ZIBinomialExpert) = d.p0
function params_init(y, d::ZIBinomialExpert)
    n_init = Int(maximum(vec(y))) + 2
    μ, σ2 = mean(y), var(y)
    p_init = ( (σ2+μ*μ)/μ - 1 ) / (n_init - 1)
    p0_init = 1 - μ/(n_init*p_init)
    try 
        return ZIBinomialExpert(p0_init, n_init, p_init)
    catch; 
        return ZIBinomialExpert() 
    end
end

## Simululation
sim_expert(d::ZIBinomialExpert, sample_size) = (1 .- Distributions.rand(Distributions.Bernoulli(d.p0), sample_size)) .* Distributions.rand(Distributions.Binomial(d.n, d.p), sample_size)

## penalty
penalty_init(d::ZIBinomialExpert) = []
no_penalty_init(d::ZIBinomialExpert) = []
penalize(d::ZIBinomialExpert, p) = 0.0

## statistics
mean(d::ZIBinomialExpert) = (1-d.p0)*mean(Distributions.Binomial(d.n, d.p))
var(d::ZIBinomialExpert) = (1-d.p0)*var(Distributions.Binomial(d.n, d.p)) + d.p0*(1-d.p0)*(mean(Distributions.Binomial(d.n, d.p)))^2
quantile(d::ZIBinomialExpert, p) = p <= d.p0 ? 0.0 : quantile(Distributions.Binomial(d.n, d.p), p-d.p0)

## EM: M-Step
function EM_M_expert(d::ZIBinomialExpert,
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
    tmp_exp = BinomialExpert(d.n, d.p)
    tmp_update = EM_M_expert(tmp_exp,
                            tl, yl, yu, tu,
                            expert_ll_pos,
                            expert_tn_pos,
                            expert_tn_bar_pos,
                            # z_e_obs, z_e_lat, k_e,
                            z_pos_e_obs, z_pos_e_lat, k_e,
                            penalty = penalty, pen_pararms_jk = pen_pararms_jk)

    return ZIBinomialExpert(p_new, tmp_update.n, tmp_update.p)

end

## EM: M-Step, exact observations
function EM_M_expert_exact(d::ZIBinomialExpert,
                    ye,
                    expert_ll_pos,
                    z_e_obs; 
                    penalty = true, pen_pararms_jk = [Inf 1.0 Inf])

    # Old parameters
    p_old = d.p0

    # Update zero probability
    z_zero_e_obs = z_e_obs .* EM_E_z_zero_obs(ye, p_old, expert_ll_pos)
    z_pos_e_obs = z_e_obs .- z_zero_e_obs
    z_zero_e_lat = 0.0
    z_pos_e_lat = 0.0
    p_new = EM_M_zero(z_zero_e_obs, z_pos_e_obs, 0.0, 0.0, 0.0)

    # Update parameters: call its positive part
    tmp_exp = BinomialExpert(d.n, d.p)
    tmp_update = EM_M_expert_exact(tmp_exp,
                            ye,
                            expert_ll_pos,
                            z_pos_e_obs;
                            penalty = penalty, pen_pararms_jk = pen_pararms_jk)

    return ZIBinomialExpert(p_new, tmp_update.n, tmp_update.p)

end