"""
    NegativeBinomialExpert(n, p)

PMF:

```math
P(X = k) = \\frac{\\Gamma(k+r)}{k! \\Gamma(r)} p^r (1 - p)^k, \\quad \\text{for } k = 0,1,2,\\ldots.
```

See also: [Negative Binomial Distribution](https://reference.wolfram.com/language/ref/NegativeBinomialDistribution.html) (Wolfram) 

"""
struct NegativeBinomialExpert{T<:Real} <: NonZIDiscreteExpert
    n::T
    p::T
    NegativeBinomialExpert{T}(n, p) where {T<:Real} = new{T}(n, p)
end

function NegativeBinomialExpert(n::T, p::T; check_args=true) where {T <: Real}
    check_args && @check_args(NegativeBinomialExpert, 0 <= p <= 1 && n > zero(n))
    return NegativeBinomialExpert{T}(n, p)
end

## Outer constructors
NegativeBinomialExpert(n::Real, p::Real) = NegativeBinomialExpert(promote(n, p)...)
NegativeBinomialExpert(n::Integer, p::Integer) = NegativeBinomialExpert(float(n), float(p))
NegativeBinomialExpert() = NegativeBinomialExpert(1, 0.5)

## Conversion
function convert(::Type{NegativeBinomialExpert{T}}, n::S, p::S) where {T <: Real, S <: Real}
    NegativeBinomialExpert(T(n), T(p))
end
function convert(::Type{NegativeBinomialExpert{T}}, d::NegativeBinomialExpert{S}) where {T <: Real, S <: Real}
    NegativeBinomialExpert(T(d.n), T(d.p), check_args=false)
end
copy(d::NegativeBinomialExpert) = NegativeBinomialExpert(d.n, d.p, check_args=false)

## Loglikelihood of Expoert
logpdf(d::NegativeBinomialExpert, x...) = isinf(x...) ? 0.0 : Distributions.logpdf.(Distributions.NegativeBinomial(d.n, d.p), x...)
pdf(d::NegativeBinomialExpert, x...) = isinf(x...) ? -Inf : Distributions.pdf.(Distributions.NegativeBinomial(d.n, d.p), x...)
logcdf(d::NegativeBinomialExpert, x...) = isinf(x...) ? 0.0 : Distributions.logcdf.(Distributions.NegativeBinomial(d.n, d.p), x...)
cdf(d::NegativeBinomialExpert, x...) = isinf(x...) ? 1.0 : Distributions.cdf.(Distributions.NegativeBinomial(d.n, d.p), x...)

## Parameters
params(d::NegativeBinomialExpert) = (d.n, d.p)
function params_init(y, d::NegativeBinomialExpert)
    μ, σ2 = mean(y), var(y)
    p_init = μ / σ2
    n_init = μ*p_init/(1-p_init)
    try 
        NegativeBinomialExpert(n_init, p_init) 
    catch; 
        NegativeBinomialExpert() 
    end
end

## Simululation
sim_expert(d::NegativeBinomialExpert, sample_size) = Distributions.rand(Distributions.NegativeBinomial(d.n, d.p), sample_size)

## penalty
penalty_init(d::NegativeBinomialExpert) = [2.0 10.0]
no_penalty_init(d::NegativeBinomialExpert) = [1.0 Inf]
penalize(d::NegativeBinomialExpert, p) = (p[1]-1)*log(d.n) - d.n/p[2]

## Misc functions for E-Step

function _sum_densy_series(d::NegativeBinomialExpert, yl, yu)
    if isinf(yu)
        series = 0:(max(yl-1, 0))
        return d.n*(1-d.p)/d.p - sum(pdf.(d, series) .* series)[1]
    else
        series = yl:yu
        return sum(pdf.(d, series) .* series)[1]
    end
end

function _int_obs_Y_raw(d::NegativeBinomialExpert, yl, yu)
    return _sum_densy_series(d, yl, yu)
end

function _int_lat_Y_raw(d::NegativeBinomialExpert, tl, tu)
    return d.n*(1-d.p)/d.p - _sum_densy_series(d, tl, tu)
end

function _sum_denslogy_series(n_new, d::NegativeBinomialExpert, yl, yu)
    upper_finite = isinf(yu) ? Distributions.quantile(NegativeBinomial(d.n, d.p), 1-1e-10) : yu
    series = yl:(max(yl, min(yu, upper_finite+1)))
    # series = yl:yu
    return sum(pdf.(d, series) .* loggamma.(series .+ n_new))[1]
end

function _int_obs_logY_raw(n_new, d::NegativeBinomialExpert, yl, yu)
    if yl == yu
        return loggamma(yl + n_new)
    else
        return _sum_denslogy_series.(n_new, d, yl, yu)
    end
end

function _int_lat_logY_raw(n_new, d::NegativeBinomialExpert, tl, tu)
    # return (tl==0 ? 0.0 : _sum_denslogy_series.(n_new, d, 0, tl)) + (isinf(tu) ? 0.0 : _sum_denslogy_series.(n_new, d, tu, Inf))
    return (tl==0 ? 0.0 : _sum_denslogy_series.(n_new, d, 0, ceil(tl)-1)) + (isinf(tu) ? 0.0 : _sum_denslogy_series.(n_new, d, floor(tu)+1, Inf))
end

function _negativebinomial_n_to_p(n, sum_term_zkz, sum_term_zkzy;
                        penalty = true, pen_pararms_jk = [])
    return 1 - sum_term_zkzy/(n*sum_term_zkz + sum_term_zkzy)
end

function _negativebinomial_optim_n(logn,
                        d_old,
                        tl, yl, yu, tu,
                        expert_ll_pos, expert_tn_pos, expert_tn_bar_pos,
                        z_e_obs, z_e_lat, k_e,
                        Y_e_obs, Y_e_lat;
                        penalty = true, pen_pararms_jk = [])
    # Optimization in log scale for unconstrained computation    
    n_tmp = exp(logn)

    # Further E-Step
    yl_yu_unique = unique_bounds(yl, yu)
    int_obs_logY_tmp = _int_obs_logY_raw.(n_tmp, d_old, yl_yu_unique[:,1], yl_yu_unique[:,2])
    logY_e_obs = exp.(-expert_ll_pos) .* int_obs_logY_tmp[match_unique_bounds(hcat(vec(yl), vec(yu)), yl_yu_unique)]
    nan2num(logY_e_obs, 0.0) # get rid of NaN

    tl_tu_unique = unique_bounds(tl, tu)
    int_lat_logY_tmp = _int_lat_logY_raw.(n_tmp, d_old, tl_tu_unique[:,1], tl_tu_unique[:,2])
    logY_e_lat = exp.(-expert_tn_bar_pos) .* int_lat_logY_tmp[match_unique_bounds(hcat(vec(tl), vec(tu)), tl_tu_unique)]
    nan2num(logY_e_lat, 0.0) # get rid of NaN

    term_zkz = z_e_obs .+ (z_e_lat .* k_e)
    term_zkz_Y = (z_e_obs .* Y_e_obs) .+ (z_e_lat .* k_e .* Y_e_lat)
    term_zkz_logY = (z_e_obs .* logY_e_obs) .+ (z_e_lat .* k_e .* logY_e_lat)

    sum_term_zkz = sum(term_zkz)[1]
    sum_term_zkzy = sum(term_zkz_Y)[1]
    sum_term_zkzlogy = sum(term_zkz_logY)[1]

    p_tmp = _negativebinomial_n_to_p(n_tmp, sum_term_zkz, sum_term_zkzy, penalty = penalty, pen_pararms_jk = pen_pararms_jk)

    obj = sum_term_zkzlogy - sum_term_zkz*loggamma(n_tmp) + sum_term_zkz*n_tmp*log(p_tmp) + sum_term_zkzy*log(1-p_tmp)
    p = penalty ? (pen_pararms_jk[1]-1)*log(d.n) - d.n/pen_pararms_jk[2] : 0.0
    return (obj + p) * (-1.0)
end

## EM: M-Step
function EM_M_expert(d::NegativeBinomialExpert,
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
    logn_new = Optim.minimizer( Optim.optimize(x -> _negativebinomial_optim_n(x, d,
                                                tl, yl, yu, tu,
                                                expert_ll_pos, expert_tn_pos, expert_tn_bar_pos,
                                                z_e_obs, z_e_lat, k_e,
                                                Y_e_obs, Y_e_lat,
                                                penalty = penalty, pen_pararms_jk = pen_pararms_jk),
                                                max(log(d.n)-2.0, 0.0), log(d.n)+2.0 )) # ,
                                                # log(d.k)-2.0, log(d.k)+2.0 )) # ,
                                                # GoldenSection() )) 
                                                # , rel_tol = 1e-8) )
    n_new = exp(logn_new)

    term_zkz = z_e_obs .+ (z_e_lat .* k_e)
    term_zkz_Y = (z_e_obs .* Y_e_obs) .+ (z_e_lat .* k_e .* Y_e_lat)

    sum_term_zkz = sum(term_zkz)[1]
    sum_term_zkzy = sum(term_zkz_Y)[1]

    p_new = _negativebinomial_n_to_p(n_new, sum_term_zkz, sum_term_zkzy, penalty = penalty, pen_pararms_jk = pen_pararms_jk)

    return NegativeBinomialExpert(n_new, p_new)

end

## EM: M-Step, exact observations
function _negativebinomial_optim_n_exact(logn,
                        d_old,
                        ye, # tl, yl, yu, tu,
                        expert_ll_pos, # expert_tn_pos, expert_tn_bar_pos,
                        z_e_obs; # , # z_e_lat, k_e,
                        # Y_e_obs, Y_e_lat;
                        penalty = true, pen_pararms_jk = [])
    # Optimization in log scale for unconstrained computation    
    n_tmp = exp(logn)

    Y_e_obs = ye

    # Further E-Step
    # yl_yu_unique = unique_bounds(yl, yu)
    # int_obs_logY_tmp = _int_obs_logY_raw.(d_old, yl_yu_unique[:,1], yl_yu_unique[:,2])
    # logY_e_obs = loggamma.(ye .+ d_old.n) # exp.(-expert_ll_pos) .* int_obs_logY_tmp[match_unique_bounds(hcat(vec(yl), vec(yu)), yl_yu_unique)]
    logY_e_obs = loggamma.(ye .+ n_tmp)
    nan2num(logY_e_obs, 0.0) # get rid of NaN

    # tl_tu_unique = unique_bounds(tl, tu)
    # int_lat_logY_tmp = _int_lat_logY_raw.(d_old, tl_tu_unique[:,1], tl_tu_unique[:,2])
    # logY_e_lat = 0.0 # exp.(-expert_tn_bar_pos) .* int_lat_logY_tmp[match_unique_bounds(hcat(vec(tl), vec(tu)), tl_tu_unique)]
    # nan2num(logY_e_lat, 0.0) # get rid of NaN

    term_zkz = z_e_obs # .+ (z_e_lat .* k_e)
    term_zkz_Y = (z_e_obs .* Y_e_obs) # .+ (z_e_lat .* k_e .* Y_e_lat)
    term_zkz_logY = (z_e_obs .* logY_e_obs) # .+ (z_e_lat .* k_e .* logY_e_lat)

    sum_term_zkz = sum(term_zkz)[1]
    sum_term_zkzy = sum(term_zkz_Y)[1]
    sum_term_zkzlogy = sum(term_zkz_logY)[1]

    p_tmp = _negativebinomial_n_to_p(n_tmp, sum_term_zkz, sum_term_zkzy, penalty = penalty, pen_pararms_jk = pen_pararms_jk)

    obj = sum_term_zkzlogy - sum_term_zkz*loggamma(n_tmp) + sum_term_zkz*n_tmp*log(p_tmp) + sum_term_zkzy*log(1-p_tmp)
    p = penalty ? (pen_pararms_jk[1]-1)*log(d.n) - d.n/pen_pararms_jk[2] : 0.0
    return (obj + p) * (-1.0)
end
function EM_M_expert_exact(d::NegativeBinomialExpert,
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
    # Y_e_lat = 0.0 # exp.(-expert_tn_bar_pos) .* int_lat_Y_tmp[match_unique_bounds(hcat(vec(tl), vec(tu)), tl_tu_unique)]
    # nan2num(Y_e_lat, 0.0) # get rid of NaN

    # Update parameters
    logn_new = Optim.minimizer( Optim.optimize(x -> _negativebinomial_optim_n_exact(x, d,
                                                ye, # tl, yl, yu, tu,
                                                expert_ll_pos, # expert_tn_pos, expert_tn_bar_pos,
                                                z_e_obs, # z_e_lat, k_e,
                                                # Y_e_obs, Y_e_lat,
                                                penalty = penalty, pen_pararms_jk = pen_pararms_jk),
                                                max(log(d.n)-2.0, 0.0), log(d.n)+2.0 )) # ,
                                                # log(d.k)-2.0, log(d.k)+2.0 )) # ,
                                                # GoldenSection() )) 
                                                # , rel_tol = 1e-8) )
    n_new = exp(logn_new)

    term_zkz = z_e_obs # .+ (z_e_lat .* k_e)
    term_zkz_Y = (z_e_obs .* Y_e_obs) # .+ (z_e_lat .* k_e .* Y_e_lat)

    sum_term_zkz = sum(term_zkz)[1]
    sum_term_zkzy = sum(term_zkz_Y)[1]

    p_new = _negativebinomial_n_to_p(n_new, sum_term_zkz, sum_term_zkzy, penalty = penalty, pen_pararms_jk = pen_pararms_jk)

    return NegativeBinomialExpert(n_new, p_new)

end