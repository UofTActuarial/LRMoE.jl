"""
    BinomialExpert(n, p)

PMF:

```math
P(X = k) = {n \\choose k}p^k(1-p)^{n-k},  \\quad \\text{ for } k = 0,1,2, \\ldots, n.
```

See also: [Binomial Distribution](https://en.wikipedia.org/wiki/Binomial_distribution) (Wikipedia) 

"""
struct BinomialExpert{T<:Real} <: NonZIDiscreteExpert
    n::Int
    p::T
    BinomialExpert{T}(n, p) where {T<:Real} = new{T}(n, p)
end

function BinomialExpert(n::Integer, p::T; check_args=true) where {T <: Real}
    check_args && @check_args(BinomialExpert, 0 <= p <= 1 && isa(n, Integer))
    return BinomialExpert{T}(n, p)
end

## Outer constructors
# BinomialExpert(n::Integer, p::Real) = BinomialExpert(n, float(p))
BinomialExpert(n::Integer, p::Integer) = BinomialExpert(n, float(p))
BinomialExpert() = BinomialExpert(2, 0.50)

## Conversion
function convert(::Type{BinomialExpert{T}}, n::Int, p::S) where {T <: Real, S <: Real}
    BinomialExpert(n, T(p))
end
function convert(::Type{BinomialExpert{T}}, d::BinomialExpert{S}) where {T <: Real, S <: Real}
    BinomialExpert(d.n, T(d.p), check_args=false)
end
copy(d::BinomialExpert) = BinomialExpert(d.n, d.p, check_args=false)

## Loglikelihood of Expoert
logpdf(d::BinomialExpert, x...) = isinf(x...) ? 0.0 : Distributions.logpdf.(Distributions.Binomial(d.n, d.p), x...)
pdf(d::BinomialExpert, x...) = isinf(x...) ? -Inf : Distributions.pdf.(Distributions.Binomial(d.n, d.p), x...)
logcdf(d::BinomialExpert, x...) = isinf(x...) ? 0.0 : Distributions.logcdf.(Distributions.Binomial(d.n, d.p), x...)
cdf(d::BinomialExpert, x...) = isinf(x...) ? 1.0 : Distributions.cdf.(Distributions.Binomial(d.n, d.p), x...)

## Parameters
params(d::BinomialExpert) = (d.n, d.p)
function params_init(y, d::BinomialExpert)
    n_init = Int(maximum(vec(y))) + 2
    p_init = mean(y) / n_init
    try 
        return BinomialExpert(n_init, p_init)
    catch; 
        return BinomialExpert() 
    end
end

## Simululation
sim_expert(d::BinomialExpert, sample_size) = Distributions.rand(Distributions.Binomial(d.n, d.p), sample_size)

## penalty
penalty_init(d::BinomialExpert) = []
no_penalty_init(d::BinomialExpert) = []
penalize(d::BinomialExpert, p) = 0.0

## statistics
mean(d::BinomialExpert) = mean(Distributions.Binomial(d.n, d.p))
var(d::BinomialExpert) = var(Distributions.Binomial(d.n, d.p))
quantile(d::BinomialExpert, p) = quantile(Distributions.Binomial(d.n, d.p), p)

## Misc functions for E-Step

function _sum_densy_series(d::BinomialExpert, yl, yu)
    if isinf(yu)
        series = 0:(max(yl-1, 0))
        return d.n*d.p - sum(pdf.(d, series) .* series)[1]
    else
        series = yl:yu
        return sum(pdf.(d, series) .* series)[1]
    end
end

function _int_obs_Y_raw(d::BinomialExpert, yl, yu)
    return _sum_densy_series(d, yl, yu)
end

function _int_lat_Y_raw(d::BinomialExpert, tl, tu)
    return d.n*d.p - _sum_densy_series(d, tl, tu)
end

## EM: M-Step
function EM_M_expert(d::BinomialExpert,
                    tl, yl, yu, tu,
                    expert_ll_pos,
                    expert_tn_pos,
                    expert_tn_bar_pos,
                    z_e_obs, z_e_lat, k_e;
                    penalty = true, pen_pararms_jk = [2.0 1.0])

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
    term_zkz_Y = (z_e_obs .* Y_e_obs) .+ (z_e_lat .* k_e .* Y_e_lat)
    term_zkz_n_Y = (z_e_obs .* (d.n .- Y_e_obs)) .+ (z_e_lat .* k_e .* (d.n .- Y_e_lat))

    p_new = sum(term_zkz_Y)[1] / (sum(term_zkz_Y)[1] + sum(term_zkz_n_Y)[1])

    return BinomialExpert(d.n, p_new)

end

## EM: M-Step, exact observations
function EM_M_expert_exact(d::BinomialExpert,
                    ye,
                    expert_ll_pos,
                    z_e_obs; 
                    penalty = true, pen_pararms_jk = [Inf 1.0 Inf])

    # Further E-Step
    # yl_yu_unique = unique_bounds(yl, yu)
    # int_obs_Y_tmp = _int_obs_Y_raw.(d, yl_yu_unique[:,1], yl_yu_unique[:,2])
    Y_e_obs = ye # exp.(-expert_ll_pos) .* int_obs_Y_tmp[match_unique_bounds(hcat(vec(yl), vec(yu)), yl_yu_unique)]
    # nan2num(Y_e_obs, 0.0) # get rid of NaN

    # tl_tu_unique = unique_bounds(tl, tu)
    # int_lat_Y_tmp = _int_lat_Y_raw.(d, tl_tu_unique[:,1], tl_tu_unique[:,2])
    Y_e_lat = 0.0 # exp.(-expert_tn_bar_pos) .* int_lat_Y_tmp[match_unique_bounds(hcat(vec(tl), vec(tu)), tl_tu_unique)]
    # nan2num(Y_e_lat, 0.0) # get rid of NaN

    # Update parameters
    term_zkz_Y = (z_e_obs .* Y_e_obs) # .+ (z_e_lat .* k_e .* Y_e_lat)
    term_zkz_n_Y = (z_e_obs .* (d.n .- Y_e_obs)) # .+ (z_e_lat .* k_e .* (d.n .- Y_e_lat))

    p_new = sum(term_zkz_Y)[1] / (sum(term_zkz_Y)[1] + sum(term_zkz_n_Y)[1])

    return BinomialExpert(d.n, p_new)

end