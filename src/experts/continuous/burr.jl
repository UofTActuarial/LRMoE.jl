"""
    BurrExpert(k, c, λ)

PDF:

```math
f(x; k, c, \\lambda) = \\frac{kc}{\\lambda} \\left( \\frac{x}{\\lambda} \\right)^{c-1} \\left( 1+ \\left( \\frac{x}{\\lambda} \\right)^{c} \\right)^{-k-1},
\\quad x \\geq 0
```

See also: [Burr Distribution](https://www.mathworks.com/help/stats/burr-type-xii-distribution.html) (Mathworks, implemented in this package),
    [Burr Distribution](https://en.wikipedia.org/wiki/Burr_distribution) (Wikipedia, with λ = 1)

"""
struct BurrExpert{T<:Real} <: NonZIContinuousExpert
    k::T
    c::T
    λ::T
    BurrExpert{T}(k::T, c::T, λ::T) where {T<:Real} = new{T}(k, c, λ)
end

function BurrExpert(k::T, c::T, λ::T; check_args=true) where {T <: Real}
    check_args && @check_args(BurrExpert, k >= zero(k) && c >= zero(c) && λ > zero(λ))
    return BurrExpert{T}(k, c, λ)
end

## Outer constructors
BurrExpert(k::Real, c::Real, λ::Real) = BurrExpert(promote(k, c, λ)...)
BurrExpert(k::Integer, c::Integer, λ::Integer) = BurrExpert(float(k), float(c), float(λ))

## Conversion
function convert(::Type{BurrExpert{T}}, k::S, c::S, λ::S) where {T <: Real, S <: Real}
    BurrExpert(T(k), T(c), T(λ))
end
function convert(::Type{BurrExpert{T}}, d::BurrExpert{S}) where {T <: Real, S <: Real}
    BurrExpert(T(d.k), T(d.c), T(d.λ), check_args=false)
end
copy(d::BurrExpert) = BurrExpert(d.k, d.c, d.λ, check_args=false)

## Loglikelihood of Expert
logpdf(d::BurrExpert, x...) = Distributions.logpdf.(LRMoE.Burr(d.k, d.c, d.λ), x...)
pdf(d::BurrExpert, x...) = Distributions.pdf.(LRMoE.Burr(d.k, d.c, d.λ), x...)
logcdf(d::BurrExpert, x...) = Distributions.logcdf.(LRMoE.Burr(d.k, d.c, d.λ), x...)
cdf(d::BurrExpert, x...) = Distributions.cdf.(LRMoE.Burr(d.k, d.c, d.λ), x...)

## Parameters
params(d::BurrExpert) = (d.k, d.c, d.λ)

## Simululation
sim_expert(d::BurrExpert, sample_size) = Distributions.rand(LRMoE.Burr(d.k, d.c, d.λ), sample_size)

## penalty
penalty_init(d::BurrExpert) = [1.0 Inf 1.0 Inf 1.0 Inf]
penalize(d::BurrExpert, p) = (p[1]-1)*log(d.k) - d.k/p[2] + (p[3]-1)*log(d.c) - d.c/p[4] + (p[5]-1)*log(d.λ) - d.λ/p[6]

## Misc functions for E-Step
function _int_logy_func(d::BurrExpert, x)
    return (iszero(x) || isinf(x)) ? 0.0 : log(x) * pdf.(d, x)
end

function _int_obs_logY_raw(d::BurrExpert, yl, yu)
    if yl == yu
        return log(yl) * pdf.(d, yl)
    else
        return quadgk.(x -> _int_logy_func(d, x), yl, yu, rtol = 1e-8)[1]
    end
end

function _int_lat_logY_raw(d::BurrExpert, tl, tu)
    return (tl==0 ? 0.0 : quadgk.(x -> _int_logy_func(d, x), 0.0, tl, rtol = 1e-8)[1]) + (isinf(tu) ? 0.0 : quadgk.(x -> _int_logy_func(d, x), tu, Inf, rtol = 1e-8)[1])
end

function _int_lpow_func(c_new, λ_new, d::BurrExpert, x)
    return (iszero(x) || isinf(x)) ? 0.0 : log1p((x/λ_new)^c_new) * pdf.(d, x)
end

function _int_obs_lpow_raw(c_new, λ_new, d::BurrExpert, yl, yu)
    if yl == yu
        return  log1p((yl/λ_new)^c_new) * pdf.(d, yl) # log(1+(yl/λ_new)^c_new) * pdf.(d, yl)
    else
        return quadgk.(x -> _int_lpow_func(c_new, λ_new, d, x), yl, yu, rtol = 1e-8)[1]
    end
end

function _int_lat_lpow_raw(c_new, λ_new, d::BurrExpert, tl, tu)
    return (tl==0 ? 0.0 : quadgk.(x -> _int_lpow_func(c_new, λ_new, d, x), 0.0, tl, rtol = 1e-8)[1]) + (isinf(tu) ? 0.0 : quadgk.(x -> _int_lpow_func(c_new, λ_new, d, x), tu, Inf, rtol = 1e-8)[1])
end

function _burr_lpow_to_k(sum_term_zkz, sum_term_zkzlpow;
    penalty = true, pen_pararms_jk = [1.0 Inf 1.0 Inf 1.0 Inf])
    if penalty
        return (sum_term_zkz + (pen_pararms_jk[1]-1))/(sum_term_zkzlpow + 1/pen_pararms_jk[2]) 
    else
        return (sum_term_zkz / sum_term_zkzlpow)
    end
end

function _burr_optim_params(lognew,
                        d_old,
                        tl, yl, yu, tu,
                        expert_ll_pos, expert_tn_pos, expert_tn_bar_pos,
                        z_e_obs, z_e_lat, k_e,
                        sum_term_zkzlogy;
                        penalty = true, pen_pararms_jk = [1.0 Inf 1.0 Inf])
    # Optimization in log scale for unconstrained computation    
    c_tmp = exp(lognew[1])
    λ_tmp = exp(lognew[2])

    yl_yu_unique = unique_bounds(yl, yu)
    int_obs_lpow_tmp = _int_obs_lpow_raw.(c_tmp, λ_tmp, d_old, yl_yu_unique[:,1], yl_yu_unique[:,2])
    lpow_e_obs = exp.(-expert_ll_pos) .* int_obs_lpow_tmp[match_unique_bounds(hcat(vec(yl), vec(yu)), yl_yu_unique)]
    nan2num(lpow_e_obs, 0.0) # get rid of NaN

    tl_tu_unique = unique_bounds(tl, tu)
    int_lat_lpow_tmp = _int_lat_lpow_raw.(c_tmp, λ_tmp, d_old, tl_tu_unique[:,1], tl_tu_unique[:,2])
    lpow_e_lat = exp.(-expert_tn_bar_pos) .* int_lat_lpow_tmp[match_unique_bounds(hcat(vec(tl), vec(tu)), tl_tu_unique)]
    nan2num(lpow_e_lat, 0.0) # get rid of NaN

    term_zkz = z_e_obs .+ (z_e_lat .* k_e)
    term_zkzlpow = (z_e_obs .* lpow_e_obs) .+ (z_e_lat .* k_e .* lpow_e_lat)

    sum_term_zkz = sum(term_zkz)[1]
    sum_term_zkzlpow = sum(term_zkzlpow)[1]

    k_tmp = _burr_lpow_to_k(sum_term_zkz, sum_term_zkzlpow, penalty = penalty, pen_pararms_jk = pen_pararms_jk)

    obj = sum_term_zkz*(log(k_tmp)+log(c_tmp)-c_tmp*log(λ_tmp)) + sum_term_zkzlogy*c_tmp - (k_tmp+1)*sum_term_zkzlpow
    p = penalty ? (pen_pararms_jk[1]-1)*log(k_tmp) - k_tmp/pen_pararms_jk[2] + (pen_pararms_jk[3]-1)*log(c_tmp) - c_tmp/pen_pararms_jk[4] + (pen_pararms_jk[5]-1)*log(λ_tmp) - λ_tmp/pen_pararms_jk[6] : 0.0
    return (obj + p) * (-1.0)
end

## EM: M-Step
function EM_M_expert(d::BurrExpert,
                     tl, yl, yu, tu,
                     expert_ll_pos,
                     expert_tn_pos,
                     expert_tn_bar_pos,
                     z_e_obs, z_e_lat, k_e;
                     penalty = true, pen_pararms_jk = [1.0 Inf 1.0 Inf 1.0 Inf])

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
    term_zkz = z_e_obs[pos_idx] .+ (z_e_lat[pos_idx] .* k_e[pos_idx])
    term_zkz_logY = (z_e_obs[pos_idx] .* logY_e_obs[pos_idx]) .+ (z_e_lat[pos_idx] .* k_e[pos_idx] .* logY_e_lat[pos_idx])

    sum_term_zkzlogy = sum(term_zkz_logY)[1]
    
    logparams_new = Optim.minimizer( Optim.optimize(x -> _burr_optim_params(x, d,
                                                tl[pos_idx], yl[pos_idx], yu[pos_idx], tu[pos_idx],
                                                expert_ll_pos[pos_idx], expert_tn_pos[pos_idx], expert_tn_bar_pos[pos_idx],
                                                z_e_obs[pos_idx], z_e_lat[pos_idx], k_e[pos_idx],
                                                sum_term_zkzlogy,
                                                penalty = penalty, pen_pararms_jk = pen_pararms_jk),
                                                [log(d.c), log(d.λ)] ))
   
    c_new = exp(logparams_new[1])
    λ_new = exp(logparams_new[2])

    # Find k_new
    # yl_yu_unique = unique_bounds(yl, yu)
    int_obs_lpow_tmp = _int_obs_lpow_raw.(c_new, λ_new, d, yl_yu_unique[:,1], yl_yu_unique[:,2])
    lpow_e_obs = exp.(-expert_ll_pos) .* int_obs_lpow_tmp[match_unique_bounds(hcat(vec(yl), vec(yu)), yl_yu_unique)]
    nan2num(lpow_e_obs, 0.0) # get rid of NaN

    # tl_tu_unique = unique_bounds(tl, tu)
    int_lat_lpow_tmp = _int_lat_lpow_raw.(c_new, λ_new, d, tl_tu_unique[:,1], tl_tu_unique[:,2])
    lpow_e_lat = exp.(-expert_tn_bar_pos) .* int_lat_lpow_tmp[match_unique_bounds(hcat(vec(tl), vec(tu)), tl_tu_unique)]
    nan2num(lpow_e_lat, 0.0) # get rid of NaN

    term_zkz = z_e_obs[pos_idx] .+ (z_e_lat[pos_idx] .* k_e[pos_idx])
    term_zkzlpow = (z_e_obs[pos_idx] .* lpow_e_obs[pos_idx]) .+ (z_e_lat[pos_idx] .* k_e[pos_idx] .* lpow_e_lat[pos_idx])

    sum_term_zkz = sum(term_zkz)[1]
    sum_term_zkzlpow = sum(term_zkzlpow)[1]

    k_new = _burr_lpow_to_k(sum_term_zkz, sum_term_zkzlpow, penalty = penalty, pen_pararms_jk = pen_pararms_jk)

    return BurrExpert(k_new, c_new, λ_new)
end

## EM: M-Step, exact observations
function _burr_lpow_to_k_exact(sum_term_zkz, sum_term_zkzlpow;
    penalty = true, pen_pararms_jk = [1.0 Inf 1.0 Inf 1.0 Inf])
    if penalty
        return (sum_term_zkz + (pen_pararms_jk[1]-1))/(sum_term_zkzlpow + 1/pen_pararms_jk[2]) 
    else
        return (sum_term_zkz / sum_term_zkzlpow)
    end
end

function _burr_optim_params_exact(lognew,
                        d_old,
                        ye,
                        expert_ll_pos, # expert_tn_pos, expert_tn_bar_pos,
                        z_e_obs, # z_e_lat, k_e,
                        sum_term_zkzlogy;
                        penalty = true, pen_pararms_jk = [1.0 Inf 1.0 Inf 1.0 Inf])
    # Optimization in log scale for unconstrained computation    
    c_tmp = exp(lognew[1])
    λ_tmp = exp(lognew[2])

    # yl_yu_unique = unique_bounds(yl, yu)
    # int_obs_lpow_tmp = _int_obs_lpow_raw.(c_tmp, λ_tmp, d, yl_yu_unique[:,1], yl_yu_unique[:,2])
    lpow_e_obs = log1p.((ye./λ_tmp).^c_tmp) # log.(1.0 .+ (ye./λ_tmp).^c_tmp) # exp.(-expert_ll_pos) .* int_obs_lpow_tmp[match_unique_bounds(hcat(vec(yl), vec(yu)), yl_yu_unique)]
    # nan2num(lpow_e_obs, 0.0) # get rid of NaN

    # tl_tu_unique = unique_bounds(tl, tu)
    # int_lat_lpow_tmp = _int_lat_lpow_raw.(c_tmp, λ_tmp, d, tl_tu_unique[:,1], tl_tu_unique[:,2])
    # lpow_e_lat = 0.0 # exp.(-expert_tn_bar_pos) .* int_lat_lpow_tmp[match_unique_bounds(hcat(vec(tl), vec(tu)), tl_tu_unique)]
    # nan2num(lpow_e_lat, 0.0) # get rid of NaN

    term_zkz = z_e_obs # .+ (z_e_lat .* k_e)
    term_zkzlpow = (z_e_obs .* lpow_e_obs) # .+ (z_e_lat .* k_e .* lpow_e_lat)

    sum_term_zkz = sum(term_zkz)[1]
    sum_term_zkzlpow = sum(term_zkzlpow)[1]

    k_tmp = _burr_lpow_to_k_exact(sum_term_zkz, sum_term_zkzlpow, penalty = penalty, pen_pararms_jk = pen_pararms_jk)

    obj = sum_term_zkz*(log(k_tmp)+log(c_tmp)-c_tmp*log(λ_tmp)) + sum_term_zkzlogy*c_tmp - (k_tmp+1)*sum_term_zkzlpow
    p = penalty ? (pen_pararms_jk[1]-1)*log(k_tmp) - k_tmp/pen_pararms_jk[2] + (pen_pararms_jk[3]-1)*log(c_tmp) - c_tmp/pen_pararms_jk[4] + (pen_pararms_jk[5]-1)*log(λ_tmp) - λ_tmp/pen_pararms_jk[6] : 0.0
    return (obj + p) * (-1.0)
end

function EM_M_expert_exact(d::BurrExpert,
                            ye,
                            expert_ll_pos,
                            z_e_obs; 
                            penalty = true, pen_pararms_jk = [Inf 1.0 Inf])

    # Further E-Step
    # yl_yu_unique = unique_bounds(yl, yu)
    # int_obs_logY_tmp = _int_obs_logY_raw.(d, yl_yu_unique[:,1], yl_yu_unique[:,2])
    logY_e_obs = log.(ye) # exp.(-expert_ll_pos) .* int_obs_logY_tmp[match_unique_bounds(hcat(vec(yl), vec(yu)), yl_yu_unique)]
    # nan2num(logY_e_obs, 0.0) # get rid of NaN

    # tl_tu_unique = unique_bounds(tl, tu)
    # int_lat_logY_tmp = _int_lat_logY_raw.(d, tl_tu_unique[:,1], tl_tu_unique[:,2])
    # logY_e_lat = exp.(-expert_tn_bar_pos) .* int_lat_logY_tmp[match_unique_bounds(hcat(vec(tl), vec(tu)), tl_tu_unique)]
    # nan2num(logY_e_lat, 0.0) # get rid of NaN
    
    # Update parameters
    pos_idx = (ye .!= 0.0)
    term_zkz = z_e_obs[pos_idx] # .+ (z_e_lat[pos_idx] .* k_e[pos_idx])
    term_zkz_logY = (z_e_obs[pos_idx] .* logY_e_obs[pos_idx]) # .+ (z_e_lat[pos_idx] .* k_e[pos_idx] .* logY_e_lat[pos_idx])

    sum_term_zkzlogy = sum(term_zkz_logY)[1]
    
    logparams_new = Optim.minimizer( Optim.optimize(x -> _burr_optim_params_exact(x, d,
                                                ye[pos_idx], # tl[pos_idx], yl[pos_idx], yu[pos_idx], tu[pos_idx],
                                                expert_ll_pos[pos_idx], # expert_tn_pos[pos_idx], expert_tn_bar_pos[pos_idx],
                                                z_e_obs[pos_idx], # z_e_lat[pos_idx], k_e[pos_idx],
                                                sum_term_zkzlogy,
                                                penalty = penalty, pen_pararms_jk = pen_pararms_jk),
                                                # [log(d.c)-2.0, log(d.λ)-2.0],
                                                # [log(d.c)+2.0, log(d.λ)+2.0],
                                                [log(d.c), log(d.λ)] ))
   
    c_new = exp(logparams_new[1])
    λ_new = exp(logparams_new[2])

    # Find k_new
    # yl_yu_unique = unique_bounds(yl, yu)
    # int_obs_lpow_tmp = _int_obs_lpow_raw.(c_new, λ_new, d, yl_yu_unique[:,1], yl_yu_unique[:,2])
    lpow_e_obs = log1p.((ye ./ λ_new).^c_new) # log.(1.0 .+(ye ./ λ_new).^c_new) # exp.(-expert_ll_pos) .* int_obs_lpow_tmp[match_unique_bounds(hcat(vec(yl), vec(yu)), yl_yu_unique)]
    nan2num(lpow_e_obs, 0.0) # get rid of NaN

    # tl_tu_unique = unique_bounds(tl, tu)
    # int_lat_lpow_tmp = _int_lat_lpow_raw.(c_new, λ_new, d, tl_tu_unique[:,1], tl_tu_unique[:,2])
    # lpow_e_lat = exp.(-expert_tn_bar_pos) .* int_lat_lpow_tmp[match_unique_bounds(hcat(vec(tl), vec(tu)), tl_tu_unique)]
    # nan2num(lpow_e_lat, 0.0) # get rid of NaN

    term_zkz = z_e_obs[pos_idx] # .+ (z_e_lat[pos_idx] .* k_e[pos_idx])
    term_zkzlpow = (z_e_obs[pos_idx] .* lpow_e_obs[pos_idx]) # .+ (z_e_lat[pos_idx] .* k_e[pos_idx] .* lpow_e_lat[pos_idx])

    sum_term_zkz = sum(term_zkz)[1]
    sum_term_zkzlpow = sum(term_zkzlpow)[1]

    k_new = _burr_lpow_to_k_exact(sum_term_zkz, sum_term_zkzlpow, penalty = penalty, pen_pararms_jk = pen_pararms_jk)

    # println("$k_new, $c_new, $λ_new")

    return BurrExpert(k_new, c_new, λ_new)
end

