"""
    PoissonExpert(λ)

PMF:

```math
P(X = k) = \\frac{\\lambda^k}{k!} e^{-\\lambda}, \\quad \\text{ for } k = 0,1,2,\\ldots.
```

See also: [Poisson Distribution](https://en.wikipedia.org/wiki/Poisson_distribution) (Wikipedia) 

"""
struct PoissonExpert{T<:Real} <: NonZIDiscreteExpert
    λ::T
    PoissonExpert{T}(λ::T) where {T<:Real} = new{T}(λ)
end

function PoissonExpert(λ::T; check_args=true) where {T <: Real}
    check_args && @check_args(PoissonExpert, λ >= zero(λ))
    return PoissonExpert{T}(λ)
end

## Outer constructors
# PoissonExpert(λ::Real) = PoissonExpert(promote(λ)...)
PoissonExpert(λ::Integer) = PoissonExpert(float(λ))

## Conversion
function convert(::Type{PoissonExpert{T}}, λ::S) where {T <: Real, S <: Real}
    PoissonExpert(T(λ))
end
function convert(::Type{PoissonExpert{T}}, d::PoissonExpert{S}) where {T <: Real, S <: Real}
    PoissonExpert(T(d.λ), check_args=false)
end
copy(d::PoissonExpert) = PoissonExpert(d.λ, check_args=false)

## Loglikelihood of Expoert
logpdf(d::PoissonExpert, x...) = isinf(x...) ? 0.0 : Distributions.logpdf.(Distributions.Poisson(d.λ), x...)
pdf(d::PoissonExpert, x...) = isinf(x...) ? -Inf : Distributions.pdf.(Distributions.Poisson(d.λ), x...)
logcdf(d::PoissonExpert, x...) = isinf(x...) ? 0.0 : Distributions.logcdf.(Distributions.Poisson(d.λ), x...)
cdf(d::PoissonExpert, x...) = isinf(x...) ? 1.0 : Distributions.cdf.(Distributions.Poisson(d.λ), x...)

## Parameters
params(d::PoissonExpert) = (d.λ)

## Simululation
sim_expert(d::PoissonExpert, sample_size) = Distributions.rand(Distributions.Poisson(d.λ), sample_size)

## penalty
penalty_init(d::PoissonExpert) = [2.0 1.0]
penalize(d::PoissonExpert, p) = (p[1]-1)*log(d.λ) - d.λ/p[2]

## Misc functions for E-Step

function _sum_densy_series(d::PoissonExpert, yl, yu)
    if isinf(yu)
        series = 0:(max(yl-1, 0))
        return d.λ - sum(pdf.(d, series) .* series)[1]
    else
        series = yl:yu
        return sum(pdf.(d, series) .* series)[1]
    end
end

function _int_obs_Y_raw(d::PoissonExpert, yl, yu)
    # if yl == yu
    #     return yl
    # else
    #     return _sum_densy_series(d, yl, yu)
    # end
    return _sum_densy_series(d, yl, yu)
end

function _int_lat_Y_raw(d::PoissonExpert, tl, tu)
    return d.λ - _sum_densy_series(d, tl, tu)
end

## EM: M-Step
function EM_M_expert(d::PoissonExpert,
                    tl, yl, yu, tu,
                    expert_ll_pos,
                    expert_tn_pos,
                    expert_tn_bar_pos,
                    z_e_obs, z_e_lat, k_e;
                    penalty = true, pen_pararms_jk = [2.0 1.0])

    # Old parameters
    λ_old = d.λ

    # Further E-Step
    yl_yu_unique = unique_bounds(yl, yu)
    int_obs_Y_tmp = _int_obs_Y_raw.(d, yl_yu_unique[:,1], yl_yu_unique[:,2])
    Y_e_obs = exp.(-expert_ll_pos) .* int_obs_Y_tmp[match_unique_bounds(hcat(vec(yl), vec(yu)), yl_yu_unique)]
    nan2num(Y_e_obs, 0.0) # get rid of NaN

    tl_tu_unique = unique_bounds(tl, tu)
    int_lat_Y_tmp = _int_lat_Y_raw.(d, tl_tu_unique[:,1], tl_tu_unique[:,2])
    Y_e_lat = exp.(-expert_tn_bar_pos) .* int_lat_Y_tmp[match_unique_bounds(hcat(vec(tl), vec(tu)), tl_tu_unique)]
    nan2num(Y_e_lat, 0.0) # get rid of NaN

    # Update parameters
    term_zkz = z_e_obs .+ (z_e_lat .* k_e)
    term_zkz_Y = (z_e_obs .* Y_e_obs) .+ (z_e_lat .* k_e .* Y_e_lat)

    λ_new = penalty ? ((sum(term_zkz_Y)[1] - (pen_pararms_jk[1]-1)) / (sum(term_zkz)[1] + 1/pen_pararms_jk[2])) : (sum(term_zkz_Y)[1] / sum(term_zkz)[1])

    return PoissonExpert(λ_new)
end

## EM: M-Step, exact observations
function EM_M_expert_exact(d::PoissonExpert,
                    ye,
                    expert_ll_pos,
                    z_e_obs; 
                    penalty = true, pen_pararms_jk = [Inf 1.0 Inf])

    # Old parameters
    λ_old = d.λ

    # Further E-Step
    # yl_yu_unique = unique_bounds(yl, yu)
    # int_obs_Y_tmp = _int_obs_Y_raw.(d, yl_yu_unique[:,1], yl_yu_unique[:,2])
    # Y_e_obs = exp.(-expert_ll_pos) .* int_obs_Y_tmp[match_unique_bounds(hcat(vec(yl), vec(yu)), yl_yu_unique)]
    # nan2num(Y_e_obs, 0.0) # get rid of NaN

    # tl_tu_unique = unique_bounds(tl, tu)
    # int_lat_Y_tmp = _int_lat_Y_raw.(d, tl_tu_unique[:,1], tl_tu_unique[:,2])
    # Y_e_lat = exp.(-expert_tn_bar_pos) .* int_lat_Y_tmp[match_unique_bounds(hcat(vec(tl), vec(tu)), tl_tu_unique)]
    # nan2num(Y_e_lat, 0.0) # get rid of NaN

    # Update parameters
    term_zkz = z_e_obs # .+ (z_e_lat .* k_e)
    term_zkz_Y = (z_e_obs .* ye) # (z_e_obs .* Y_e_obs) .+ (z_e_lat .* k_e .* Y_e_lat)

    λ_new = penalty ? ((sum(term_zkz_Y)[1] - (pen_pararms_jk[1]-1)) / (sum(term_zkz)[1] + 1/pen_pararms_jk[2])) : (sum(term_zkz_Y)[1] / sum(term_zkz)[1])

    return PoissonExpert(λ_new)
end