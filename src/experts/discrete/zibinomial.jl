"""
    ZIBinomialExpert(p0, n, p)

Expert function: `ZIBinomialExpert(p0, n, p)`.

"""
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

## Parameters
params(d::ZIBinomialExpert) = (d.p0, d.n, d.p)

## Simululation
sim_expert(d::ZIBinomialExpert, sample_size) = (1 .- Distributions.rand(Distributions.Bernoulli(d.p0), sample_size)) .* Distributions.rand(Distributions.Binomial(d.n, d.p), sample_size)

## penalty
penalty_init(d::ZIBinomialExpert) = []
penalize(d::ZIBinomialExpert, p) = 0.0

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