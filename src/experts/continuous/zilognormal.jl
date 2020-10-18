"""
    ZILogNormalExpert(p, μ, σ)

Expert function: `ZILogNormalExpert(p, μ, σ)`.

"""
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
penalize(d::ZILogNormalExpert, p) = (d.μ/p[1])^2 + (p[2]-1)*log(d.σ) - d.σ/p[3]

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
                            z_e_obs, z_e_lat, k_e,
                            penalty = penalty, pen_pararms_jk = pen_pararms_jk)

    return ZILogNormalExpert(p_new, tmp_update.μ, tmp_update.σ)
end