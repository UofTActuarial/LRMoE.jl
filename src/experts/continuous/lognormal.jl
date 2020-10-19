"""
    LogNormalExpert(μ, σ)

Expert function: `LogNormal(μ, σ)`.

"""
struct LogNormalExpert{T<:Real} <: NonZIContinuousExpert
    μ::T
    σ::T
    LogNormalExpert{T}(µ::T, σ::T) where {T<:Real} = new{T}(µ, σ)
end

function LogNormalExpert(μ::T, σ::T; check_args=true) where {T <: Real}
    check_args && @check_args(LogNormalExpert, σ >= zero(σ))
    return LogNormalExpert{T}(μ, σ)
end

## Outer constructors
LogNormalExpert(μ::Real, σ::Real) = LogNormalExpert(promote(μ, σ)...)
LogNormalExpert(μ::Integer, σ::Integer) = LogNormalExpert(float(μ), float(σ))

## Conversion
function convert(::Type{LogNormalExpert{T}}, μ::S, σ::S) where {T <: Real, S <: Real}
    LogNormalExpert(T(μ), T(σ))
end
function convert(::Type{LogNormalExpert{T}}, d::LogNormalExpert{S}) where {T <: Real, S <: Real}
    LogNormalExpert(T(d.μ), T(d.σ), check_args=false)
end
copy(d::LogNormalExpert) = LogNormalExpert(d.μ, d.σ, check_args=false)

## Loglikelihood of Expert
logpdf(d::LogNormalExpert, x...) = Distributions.logpdf.(Distributions.LogNormal(d.μ, d.σ), x...)
pdf(d::LogNormalExpert, x...) = Distributions.pdf.(Distributions.LogNormal(d.μ, d.σ), x...)
logcdf(d::LogNormalExpert, x...) = Distributions.logcdf.(Distributions.LogNormal(d.μ, d.σ), x...)
cdf(d::LogNormalExpert, x...) = Distributions.cdf.(Distributions.LogNormal(d.μ, d.σ), x...)

## Parameters
params(d::LogNormalExpert) = (d.μ, d.σ)

## Simululation
sim_expert(d::LogNormalExpert, sample_size) = Distributions.rand(Distributions.LogNormal(d.μ, d.σ), sample_size)

## penalty
penalty_init(d::LogNormalExpert) = [Inf 1.0 Inf]
penalize(d::LogNormalExpert, p) = (d.μ/p[1])^2 + (p[2]-1)*log(d.σ) - d.σ/p[3]

## Misc functions for E-Step
function _diff_dens_series(d::LogNormalExpert, yl, yu)
    return exp(-0.5*(log(yl)-d.μ)^2/(d.σ^2)) - exp(-0.5*(log(yu)-d.μ)^2/(d.σ^2))
end

function _diff_dist_series(d::LogNormalExpert, yl, yu)
    return ( 0.5 + 0.5*erf((log(yu)-d.μ)/(sqrt2*d.σ)) ) - ( 0.5 + 0.5*erf((log(yl)-d.μ)/(sqrt2*d.σ)) )
end

function int_obs_logY(d::LogNormalExpert, yl, yu, expert_ll)
    if yl == yu
        return log(yl)
    else
        return exp(-expert_ll) * ( d.σ*invsqrt2π*0.5*_diff_dens_series(d, yl, yu) + d.μ*_diff_dist_series(d, yl, yu) )
    end
end

function int_lat_logY(d::LogNormalExpert, tl, tu, expert_tn_bar)
    return exp(-expert_tn_bar) * (d.μ - ( d.σ*invsqrt2π*0.5*_diff_dens_series(d, tl, tu) + d.μ*_diff_dist_series(d, tl, tu) ) )
end

function _zdensz_series(z)
    return (isinf(z) ? 0.0 : z*exp(-0.5*z^2))
end

function _diff_zdensz_series(d::LogNormalExpert, yl, yu)
    return _zdensz_series((log(yl)-d.μ)/d.σ) - _zdensz_series((log(yu)-d.μ)/d.σ)
end

function int_obs_logY_sq(d::LogNormalExpert, yl, yu, expert_ll)
    if yl == yu
        return (log(yl))^2
    else
        return exp(-expert_ll) * ( ((d.σ)^2 + (d.μ)^2)*_diff_dist_series(d, yl, yu) + d.σ*d.μ*invsqrt2π*_diff_dens_series(d, yl, yu) + (d.σ)^2*invsqrt2π*_diff_zdensz_series(d, yl, yu) )
    end
end

function int_lat_logY_sq(d::LogNormalExpert, tl, tu, expert_tn_bar)
    return exp(-expert_tn_bar) * ( ((d.σ)^2 + (d.μ)^2) - ( ((d.σ)^2 + (d.μ)^2)*_diff_dist_series(d, tl, tu) + d.σ*d.μ*invsqrt2π*_diff_dens_series(d, tl, tu) + (d.σ)^2*invsqrt2π*_diff_zdensz_series(d, tl, tu) ))
end

## EM: M-Step
function EM_M_expert(d::LogNormalExpert,
                     tl, yl, yu, tu,
                     expert_ll_pos,
                     expert_tn_pos,
                     expert_tn_bar_pos,
                     z_e_obs, z_e_lat, k_e;
                     penalty = true, pen_pararms_jk = [Inf 1.0 Inf])

    # Old parameters
    μ_old = d.μ
    σ_old = d.σ

    # println("$(μ_old), $(σ_old)")

    # Further E-Step
    logY_e_obs = vec( int_obs_logY.(d, yl, yu, expert_ll_pos) )
    logY_e_lat = vec( int_lat_logY.(d, tl, tu, expert_tn_bar_pos) )
    nan2num(logY_e_lat, 0.0) # get rid of NaN
    
    logY_sq_e_obs = vec( int_obs_logY_sq.(d, yl, yu, expert_ll_pos) )
    logY_sq_e_lat = vec( int_lat_logY_sq.(d, tl, tu, expert_tn_bar_pos) )
    nan2num(logY_sq_e_lat, 0.0) # get rid of NaN
    
    # Update parameters
    pos_idx = (yu .!= 0.0)
    term_zkz = z_e_obs[pos_idx] .+ (z_e_lat[pos_idx] .* k_e[pos_idx])
    term_zkz_logY = (z_e_obs[pos_idx] .* logY_e_obs[pos_idx]) .+ (z_e_lat[pos_idx] .* k_e[pos_idx] .* logY_e_lat[pos_idx])
    term_zkz_logY_sq = (z_e_obs[pos_idx] .* logY_sq_e_obs[pos_idx]) .+ (z_e_lat[pos_idx] .* k_e[pos_idx] .* logY_sq_e_lat[pos_idx])

    μ_new = sum(term_zkz_logY)[1] / sum(term_zkz)[1]
    σ_new = sqrt( 1/sum(term_zkz)[1] * (sum(term_zkz_logY_sq)[1] - 2.0*μ_new*sum(term_zkz_logY)[1] + (μ_new)^2*sum(term_zkz)[1] ) )

    # println("$(μ_new), $(σ_new)")

    return LogNormalExpert(μ_new, σ_new)
end