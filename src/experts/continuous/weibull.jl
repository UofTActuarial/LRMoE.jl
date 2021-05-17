"""
    WeibullExpert(k, θ)

PDF:

```math
f(x; k, \\theta) = \\frac{k}{\\theta} \\left( \\frac{x}{\\theta} \\right)^{k-1} e^{-(x/\\theta)^k},
\\quad x \\geq 0
```

See also: [Weibull Distribution](https://en.wikipedia.org/wiki/Weibull_distribution) (Wikipedia) 

"""
struct WeibullExpert{T<:Real} <: NonZIContinuousExpert
    k::T
    θ::T
    WeibullExpert{T}(k::T, θ::T) where {T<:Real} = new{T}(k, θ)
end

function WeibullExpert(k::T, θ::T; check_args=true) where {T <: Real}
    check_args && @check_args(WeibullExpert, k >= one(k) && θ > zero(θ))
    return WeibullExpert{T}(k, θ)
end

## Outer constructors
WeibullExpert(k::Real, θ::Real) = WeibullExpert(promote(k, θ)...)
WeibullExpert(k::Integer, θ::Integer) = WeibullExpert(float(k), float(θ))
WeibullExpert() = WeibullExpert(2.0, 1.0)

## Conversion
function convert(::Type{WeibullExpert{T}}, k::S, θ::S) where {T <: Real, S <: Real}
    WeibullExpert(T(k), T(θ))
end
function convert(::Type{WeibullExpert{T}}, d::WeibullExpert{S}) where {T <: Real, S <: Real}
    WeibullExpert(T(d.k), T(d.θ), check_args=false)
end
copy(d::WeibullExpert) = WeibullExpert(d.k, d.θ, check_args=false)

## Loglikelihood of Expert
logpdf(d::WeibullExpert, x...) = Distributions.logpdf.(Distributions.Weibull(d.k, d.θ), x...)
pdf(d::WeibullExpert, x...) = Distributions.pdf.(Distributions.Weibull(d.k, d.θ), x...)
logcdf(d::WeibullExpert, x...) = Distributions.logcdf.(Distributions.Weibull(d.k, d.θ), x...)
cdf(d::WeibullExpert, x...) = Distributions.cdf.(Distributions.Weibull(d.k, d.θ), x...)

## expert_ll, etc
expert_ll_exact(d::WeibullExpert, x::Real; exposure = 1) = LRMoE.logpdf(d, x) 
function expert_ll(d::WeibullExpert, tl::Real, yl::Real, yu::Real, tu::Real; exposure = 1)
    expert_ll = (yl == yu) ? logpdf.(d, yl) : logcdf.(d, yu) + log1mexp.(logcdf.(d, yl) - logcdf.(d, yu))
    expert_ll = (tu == 0.) ? -Inf : expert_ll
    return expert_ll
end
function expert_tn(d::WeibullExpert, tl::Real, yl::Real, yu::Real, tu::Real; exposure = 1)
    expert_tn = (tl == tu) ? logpdf.(d, tl) : logcdf.(d, tu) + log1mexp.(logcdf.(d, tl) - logcdf.(d, tu))
    expert_tn = (tu == 0.) ? -Inf : expert_tn
    return expert_tn
end
function expert_tn_bar(d::WeibullExpert, tl::Real, yl::Real, yu::Real, tu::Real; exposure = 1)
    expert_tn_bar = (tl == tu) ? 0.0 : log1mexp.(logcdf.(d, tu) + log1mexp.(logcdf.(d, tl) - logcdf.(d, tu)))
    return expert_tn_bar
end

## Parameters
params(d::WeibullExpert) = (d.k, d.θ)
function params_init(y, d::WeibullExpert)
    pos_idx = (y .> 0.0)
    
    function init_obj(logk, y)
        n = length(y)
        k_tmp = exp(logk)
        θ_tmp = ( sum(y.^k_tmp) / n)^(1/k_tmp)
        return -1 * (n*logk - n*log(θ_tmp) + k_tmp*sum(log.(y)) - n*(k_tmp-1)*log(θ_tmp) - 1/(θ_tmp^k_tmp)*sum(y.^k_tmp) )
    end
    
    logk_init = Optim.minimizer( Optim.optimize(x -> init_obj(x, y[pos_idx]),  0.0, 2.0 ))
    
    k_init = exp(logk_init)
    θ_init = ( sum(y[pos_idx].^k_init) / length(y[pos_idx]))^(1/k_init)

    try 
        return WeibullExpert(k_init, θ_init)
    catch; 
        WeibullExpert() 
    end
end

## KS stats for parameter initialization
function ks_distance(y, d::WeibullExpert)
    p_zero = sum(y .== 0.0) / sum(y .>= 0.0)
    return max(abs(p_zero-0.0), (1-0.0)*HypothesisTests.ksstats(y[y .> 0.0], Distributions.Weibull(d.k, d.θ))[2])
end


## Simululation
sim_expert(d::WeibullExpert, sample_size) = Distributions.rand(Distributions.Weibull(d.k, d.θ), sample_size)

## penalty
penalty_init(d::WeibullExpert) = [2.0 10.0 1.0 Inf]
no_penalty_init(d::WeibullExpert) = [1.0 Inf 1.0 Inf]
penalize(d::WeibullExpert, p) = (p[1]-1)*log(d.k) - d.k/p[2] + (p[3]-1)*log(d.θ) - d.θ/p[4]

## statistics
mean(d::WeibullExpert) = mean(Distributions.Weibull(d.k, d.θ))
var(d::WeibullExpert) = var(Distributions.Weibull(d.k, d.θ))
quantile(d::WeibullExpert, p) = quantile(Distributions.Weibull(d.k, d.θ), p)
lev(d::WeibullExpert, u) = d.θ*gamma_inc(float(1/d.k+1), (u/d.θ)^d.k, 0)[1]*gamma(float(1/d.k+1)) + u*exp(-(u/d.θ)^d.k)
excess(d::WeibullExpert, u) = mean(d) - lev(d, u)

## Misc functions for E-Step

function _int_logy_func(d::WeibullExpert, x)
    return (iszero(x) || isinf(x)) ? 0.0 : log(x) * pdf.(d, x)
end

function _int_obs_logY_raw(d::WeibullExpert, yl, yu)
    if yl == yu
        return log(yl) * pdf.(d, yl)
    else
        return quadgk.(x -> _int_logy_func(d, x), yl, yu, rtol = 1e-8)[1]
    end
end

function _int_lat_logY_raw(d::WeibullExpert, tl, tu)
    return (tl==0 ? 0.0 : quadgk.(x -> _int_logy_func(d, x), 0.0, tl, rtol = 1e-8)[1]) + (isinf(tu) ? 0.0 : quadgk.(x -> _int_logy_func(d, x), tu, Inf, rtol = 1e-8)[1])
end

function _int_powy_func(d_old::WeibullExpert, k_new, l, u)
    term1 = isinf(l) ? 0.0 : gamma((d_old.k+k_new)/d_old.k) * gamma_inc((d_old.k+k_new)/d_old.k, (l/d_old.θ)^(d_old.k), 1)[2]
    term2 = isinf(u) ? 0.0 : gamma((d_old.k+k_new)/d_old.k) * gamma_inc((d_old.k+k_new)/d_old.k, (u/d_old.θ)^(d_old.k), 1)[2]
    return (d_old.θ)^(k_new) * (term1 - term2)
end

function _int_obs_powY(d_old::WeibullExpert, k_new, yl, yu, expert_ll_pos)
    if yl == yu
        return yl^k_new
    else
        return exp(-expert_ll_pos) * _int_powy_func(d_old, k_new, yl, yu)
    end
end

function _int_lat_powY(d_old::WeibullExpert, k_new, tl, tu, expert_tn_bar_pos)
    return exp(-expert_tn_bar_pos) * (_int_powy_func(d_old, k_new, 0.0, tl) + _int_powy_func(d_old, k_new, tu, Inf))
end


function _weibull_k_to_λ(k, sum_term_zkz, sum_term_zkzpowy;
                        penalty = true, pen_pararms_jk = [1.0 Inf 1.0 Inf])

    return (sum_term_zkzpowy / sum_term_zkz)^(1/k)
end

function _weibull_optim_k(logk,
                        d_old,
                        tl, yl, yu, tu,
                        expert_ll_pos, expert_tn_pos, expert_tn_bar_pos,
                        z_e_obs, z_e_lat, k_e,
                        logY_e_obs, logY_e_lat;
                        penalty = true, pen_pararms_jk = [1.0 Inf 1.0 Inf])
    # Optimization in log scale for unconstrained computation    
    k_tmp = exp(logk)

    # Further E-step
    powY_e_obs = _int_obs_powY.(d_old, k_tmp, yl, yu, expert_ll_pos)
    powY_e_lat = _int_lat_powY.(d_old, k_tmp, tl, tu, expert_tn_bar_pos)
    nan2num(powY_e_obs, 0.0) # get rid of NaN
    nan2num(powY_e_lat, 0.0) # get rid of NaN

    term_zkz = z_e_obs .+ (z_e_lat .* k_e)
    term_zkz_logY = (z_e_obs .* logY_e_obs) .+ (z_e_lat .* k_e .* logY_e_lat)
    term_zkz_powY = (z_e_obs .* powY_e_obs) .+ (z_e_lat .* k_e .* powY_e_lat)

    sum_term_zkz = sum(term_zkz)[1]
    sum_term_zkz_logY = sum(term_zkz_logY)[1]
    sum_term_zkz_powY = sum(term_zkz_powY)[1]

    θ_tmp = _weibull_k_to_λ(k_tmp, sum_term_zkz, sum_term_zkz_powY, penalty = penalty, pen_pararms_jk = pen_pararms_jk)

    obj = log(k_tmp)*sum_term_zkz - k_tmp*log(θ_tmp)*sum_term_zkz + (k_tmp-1)*sum_term_zkz_logY - (θ_tmp^(-k_tmp))*sum_term_zkz_powY
    p = penalty ? (pen_pararms_jk[1]-1)*log(k_tmp) + k_tmp/pen_pararms_jk[2] + (pen_pararms_jk[3]-1)*log(θ_tmp) + θ_tmp/pen_pararms_jk[4] : 0.0
    return (obj + p) * (-1.0)

end

## EM: M-Step
function EM_M_expert(d::WeibullExpert,
                     tl, yl, yu, tu,
                     expert_ll_pos,
                     expert_tn_pos,
                     expert_tn_bar_pos,
                     z_e_obs, z_e_lat, k_e;
                     penalty = true, pen_pararms_jk = [1.0 Inf 1.0 Inf])

    # Further E-Step
    yl_yu_unique = unique_bounds(yl, yu)
    int_obs_logY_tmp = _int_obs_logY_raw.(d, yl_yu_unique[:,1], yl_yu_unique[:,2])
    logY_e_obs = exp.(-expert_ll_pos) .* int_obs_logY_tmp[match_unique_bounds(hcat(vec(yl), vec(yu)), yl_yu_unique)]
    nan2num(logY_e_obs, 0.0) # get rid of NaN

    tl_tu_unique = unique_bounds(tl, tu)
    int_lat_logY_tmp = _int_lat_logY_raw.(d, tl_tu_unique[:,1], tl_tu_unique[:,2])
    logY_e_lat = exp.(-expert_tn_bar_pos) .* int_lat_logY_tmp[match_unique_bounds(hcat(vec(tl), vec(tu)), tl_tu_unique)]
    nan2num(logY_e_lat, 0.0) # get rid of NaN

    # Update parameters
    pos_idx = (yu .!= 0.0)
    logk_new = Optim.minimizer( Optim.optimize(x -> _weibull_optim_k(x, d,
                                                            tl[pos_idx], yl[pos_idx], yu[pos_idx], tu[pos_idx],
                                                            expert_ll_pos[pos_idx], expert_tn_pos[pos_idx], expert_tn_bar_pos[pos_idx],
                                                            z_e_obs[pos_idx], z_e_lat[pos_idx], k_e[pos_idx],
                                                            logY_e_obs[pos_idx], logY_e_lat[pos_idx],
                                                            penalty = penalty, pen_pararms_jk = pen_pararms_jk),
                                               max(log(d.k)-2.0, 0.0), log(d.k)+2.0 )) # ,
                                               # log(d.k)-2.0, log(d.k)+2.0 )) # ,
                                               # GoldenSection() )) 
                                               # , rel_tol = 1e-8) )
    k_new = exp(logk_new)

    # Further E-step
    powY_e_obs = _int_obs_powY.(d, k_new, yl, yu, expert_ll_pos)
    powY_e_lat = _int_lat_powY.(d, k_new, tl, tu, expert_tn_bar_pos)
    nan2num(powY_e_obs, 0.0) # get rid of NaN
    nan2num(powY_e_lat, 0.0) # get rid of NaN

    term_zkz = z_e_obs .+ (z_e_lat .* k_e)
    # term_zkz_logY = (z_e_obs .* logY_e_obs) .+ (z_e_lat .* k_e .* logY_e_lat)
    term_zkz_powY = (z_e_obs .* powY_e_obs) .+ (z_e_lat .* k_e .* powY_e_lat)

    sum_term_zkz = sum(term_zkz)[1]
    # sum_term_zkz_logY = sum(term_zkz_logY)[1]
    sum_term_zkz_powY = sum(term_zkz_powY)[1]

    θ_new = _weibull_k_to_λ(k_new, sum(term_zkz)[1], sum(sum_term_zkz_powY)[1], penalty = penalty, pen_pararms_jk = pen_pararms_jk)

    # println("$k_new, $θ_new")
    return WeibullExpert(k_new, θ_new)

end

## EM: M-Step, exact observations
function _weibull_optim_k_exact(logk,
                        # d_old,
                        ye,
                        # expert_ll_pos, expert_tn_pos, expert_tn_bar_pos,
                        z_e_obs, # z_e_lat, k_e,
                        logY_e_obs; #, logY_e_lat;
                        penalty = true, pen_pararms_jk = [1.0 Inf 1.0 Inf])
    # Optimization in log scale for unconstrained computation    
    k_tmp = exp(logk)

    # Further E-step
    powY_e_obs = ye.^(k_tmp) # _int_obs_powY(d_old, k_tmp, yl, yu, expert_ll_pos)
    # powY_e_lat = 0.0 # _int_lat_powY(d_old, k_tmp, tl, tu, expert_tn_bar_pos)
    nan2num(powY_e_obs, 0.0) # get rid of NaN
    # nan2num(powY_e_lat, 0.0) # get rid of NaN

    term_zkz = z_e_obs # .+ (z_e_lat .* k_e)
    term_zkz_logY = (z_e_obs .* logY_e_obs) # .+ (z_e_lat .* k_e .* logY_e_lat)
    term_zkz_powY = (z_e_obs .* powY_e_obs) # .+ (z_e_lat .* k_e .* powY_e_lat)

    sum_term_zkz = sum(term_zkz)[1]
    sum_term_zkz_logY = sum(term_zkz_logY)[1]
    sum_term_zkz_powY = sum(term_zkz_powY)[1]

    θ_tmp = _weibull_k_to_λ(k_tmp, sum_term_zkz, sum_term_zkz_powY, penalty = penalty, pen_pararms_jk = pen_pararms_jk)

    obj = log(k_tmp)*sum_term_zkz - k_tmp*log(θ_tmp)*sum_term_zkz + (k_tmp-1)*sum_term_zkz_logY - (θ_tmp^(-k_tmp))*sum_term_zkz_powY
    p = penalty ? (pen_pararms_jk[1]-1)*log(k_tmp) + k_tmp/pen_pararms_jk[2] + (pen_pararms_jk[3]-1)*log(θ_tmp) + θ_tmp/pen_pararms_jk[4] : 0.0
    return (obj + p) * (-1.0)

end
function EM_M_expert_exact(d::WeibullExpert,
                            ye,
                            expert_ll_pos,
                            z_e_obs; 
                            penalty = true, pen_pararms_jk = [Inf 1.0 Inf])

    # Further E-Step
    # yl_yu_unique = unique_bounds(yl, yu)
    # int_obs_logY_tmp = _int_obs_logY_raw.(d, yl_yu_unique[:,1], yl_yu_unique[:,2])
    logY_e_obs = log.(ye) # exp.(-expert_ll_pos) .* int_obs_logY_tmp[match_unique_bounds(hcat(vec(yl), vec(yu)), yl_yu_unique)]
    nan2num(logY_e_obs, 0.0) # get rid of NaN

    # tl_tu_unique = unique_bounds(tl, tu)
    # int_lat_logY_tmp = _int_lat_logY_raw.(d, tl_tu_unique[:,1], tl_tu_unique[:,2])
    logY_e_lat = 0.0 # exp.(-expert_tn_bar_pos) .* int_lat_logY_tmp[match_unique_bounds(hcat(vec(tl), vec(tu)), tl_tu_unique)]
    # nan2num(logY_e_lat, 0.0) # get rid of NaN
    
    # Update parameters
    pos_idx = (ye .!= 0.0)
    logk_new = Optim.minimizer( Optim.optimize(x -> _weibull_optim_k_exact(x, # d,
                                                            ye[pos_idx],
                                                            # expert_ll_pos[pos_idx], expert_tn_pos[pos_idx], expert_tn_bar_pos[pos_idx],
                                                            z_e_obs[pos_idx], # z_e_lat[pos_idx], k_e[pos_idx],
                                                            logY_e_obs[pos_idx], # logY_e_lat[pos_idx],
                                                            penalty = penalty, pen_pararms_jk = pen_pararms_jk),
                                               max(log(d.k)-2.0, 0.0), log(d.k)+2.0 )) # ,
                                               # log(d.k)-2.0, log(d.k)+2.0 )) # ,
                                               # GoldenSection() )) 
                                               # , rel_tol = 1e-8) )
    k_new = exp(logk_new)

    # Further E-step
    powY_e_obs = ye.^(k_new) # _int_obs_powY(d_old, k_tmp, yl, yu, expert_ll_pos)
    # powY_e_lat = 0.0 # _int_lat_powY(d_old, k_tmp, tl, tu, expert_tn_bar_pos)
    nan2num(powY_e_obs, 0.0) # get rid of NaN
    # nan2num(powY_e_lat, 0.0) # get rid of NaN

    term_zkz = z_e_obs # .+ (z_e_lat .* k_e)
    # term_zkz_logY = (z_e_obs .* logY_e_obs) # .+ (z_e_lat .* k_e .* logY_e_lat)
    term_zkz_powY = (z_e_obs .* powY_e_obs) # .+ (z_e_lat .* k_e .* powY_e_lat)

    sum_term_zkz = sum(term_zkz)[1]
    # sum_term_zkz_logY = sum(term_zkz_logY)[1]
    sum_term_zkz_powY = sum(term_zkz_powY)[1]

    θ_new = _weibull_k_to_λ(k_new, sum(term_zkz)[1], sum(sum_term_zkz_powY)[1], penalty = penalty, pen_pararms_jk = pen_pararms_jk)

    # println("$k_new, $θ_new")
    return WeibullExpert(k_new, θ_new)

end

