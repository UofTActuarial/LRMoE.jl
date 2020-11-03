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
ZINegativeBinomialExpert(p0::Integer, n::Integer, p::Integer) = ZINegativeBinomialExpert(float(p0), n, float(p))

## Conversion
function convert(::Type{ZINegativeBinomialExpert{T}}, p0::S, n::Int, p::S) where {T <: Real, S <: Real}
    ZINegativeBinomialExpert(T(p0), T(n), T(p))
end
function convert(::Type{ZINegativeBinomialExpert{T}}, d::ZINegativeBinomialExpert{S}) where {T <: Real, S <: Real}
    ZINegativeBinomialExpert(T(d.p0), T(d.n), T(d.p), check_args=false)
end
copy(d::ZINegativeBinomialExpert) = ZINegativeBinomialExpert(d.p0, d.n, d.p, check_args=false)

## Loglikelihood of Expoert
logpdf(d::ZINegativeBinomialExpert, x...) = isinf(x...) ? 0.0 : Distributions.logpdf.(Distributions.NegativeBinomial(d.n, d.p), x...)
pdf(d::ZINegativeBinomialExpert, x...) = isinf(x...) ? -Inf : Distributions.pdf.(Distributions.NegativeBinomial(d.n, d.p), x...)
logcdf(d::ZINegativeBinomialExpert, x...) = isinf(x...) ? 0.0 : Distributions.logcdf.(Distributions.NegativeBinomial(d.n, d.p), x...)
cdf(d::ZINegativeBinomialExpert, x...) = isinf(x...) ? 1.0 : Distributions.cdf.(Distributions.NegativeBinomial(d.n, d.p), x...)

## Parameters
params(d::ZINegativeBinomialExpert) = (d.p0, d.n, d.p)

## Simululation
sim_expert(d::ZINegativeBinomialExpert, sample_size) = (1 .- Distributions.rand(Distributions.Bernoulli(d.p0), sample_size)) .* Distributions.rand(Distributions.NegativeBinomial(d.n, d.p), sample_size)

## penalty
penalty_init(d::ZINegativeBinomialExpert) = [1.0 Inf]
penalize(d::ZINegativeBinomialExpert, p) = (p[1]-1)*log(d.n) - d.n/p[2]

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
    tmp_exp = NegativeBinomialExpert(d.n, d.p)
    tmp_update = EM_M_expert_exact(tmp_exp,
                            ye,
                            expert_ll_pos,
                            z_pos_e_obs;
                            penalty = penalty, pen_pararms_jk = pen_pararms_jk)

    return ZINegativeBinomialExpert(p_new, tmp_update.n, tmp_update.p)

end