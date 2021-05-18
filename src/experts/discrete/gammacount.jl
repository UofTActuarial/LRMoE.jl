"""
    GammaCountExpert(m, s)

PMF:

```math
P(X = k) = G(m k, s T) - G(m (k+1), s T), \\quad \\text{ for } k = 0,1,2, \\ldots, n.
```

with

```math
G(m k, s T) = \\frac{1}{\\Gamma(mk)}  \\int^{sT}_{0} u^{mk - 1} e^{-u} du
```

See also: [Gamma Count Distribution](https://arxiv.org/abs/1312.2423) (Arxiv) 

"""
struct GammaCountExpert{T<:Real} <: NonZIDiscreteExpert
    m::T
    s::T
    GammaCountExpert{T}(m, s) where {T<:Real} = new{T}(m, s)
end

function GammaCountExpert(m::T, s::T; check_args=true) where {T <: Real}
    check_args && @check_args(GammaCountExpert, m > zero(m) && s > zero(s))
    return GammaCountExpert{T}(m, s)
end

## Outer constructors
GammaCountExpert(m::Real, s::Real) = GammaCountExpert(promote(m, s)...)
GammaCountExpert(m::Integer, s::Integer) = GammaCountExpert(float(m), float(s))
GammaCountExpert() = GammaCountExpert(2.0, 1.0)

## Conversion
function convert(::Type{GammaCountExpert{T}}, m::S, s::S) where {T <: Real, S <: Real}
    GammaCountExpert(T(m), T(s))
end
function convert(::Type{GammaCountExpert{T}}, d::GammaCountExpert{S}) where {T <: Real, S <: Real}
    GammaCountExpert(T(d.m), T(d.s), check_args=false)
end
copy(d::GammaCountExpert) = GammaCountExpert(d.m, d.s, check_args=false)

## Loglikelihood of Expoert
logpdf(d::GammaCountExpert, x...) = isinf(x...) ? -Inf : LRMoE.logpdf.(LRMoE.GammaCount(d.m, d.s), x...)
pdf(d::GammaCountExpert, x...) = isinf(x...) ? 0.0 : LRMoE.pdf.(LRMoE.GammaCount(d.m, d.s), x...)
logcdf(d::GammaCountExpert, x...) = isinf(x...) ? 0.0 : LRMoE.logcdf.(LRMoE.GammaCount(d.m, d.s), x...)
cdf(d::GammaCountExpert, x...) = isinf(x...) ? 1.0 : LRMoE.cdf.(LRMoE.GammaCount(d.m, d.s), x...)

## expert_ll, etc
expert_ll_exact(d::GammaCountExpert, x::Real; exposure = 1) = LRMoE.logpdf(LRMoE.GammaCount(d.m, d.s/exposure), x) 
function expert_ll(d::GammaCountExpert, tl::Real, yl::Real, yu::Real, tu::Real; exposure = 1)
    d_exp = LRMoE.GammaCount(d.m, d.s/exposure)
    expert_ll = (yl == yu) ? logpdf.(d_exp, yl) : logcdf.(d_exp, yu) + log1mexp.(logcdf.(d_exp, ceil.(yl) .- 1) - logcdf.(d_exp, yu))
    return expert_ll
end
function expert_tn(d::GammaCountExpert, tl::Real, yl::Real, yu::Real, tu::Real; exposure = 1)
    d_exp = LRMoE.GammaCount(d.m, d.s/exposure)
    expert_tn = (tl == tu) ? logpdf.(d_exp, tl) : logcdf.(d_exp, tu) + log1mexp.(logcdf.(d_exp, ceil.(tl) .- 1) - logcdf.(d_exp, tu))
    return expert_tn
end
function expert_tn_bar(d::GammaCountExpert, tl::Real, yl::Real, yu::Real, tu::Real; exposure = 1)
    d_exp = LRMoE.GammaCount(d.m, d.s/exposure)
    expert_tn_bar = (tl == tu) ? log1mexp.(logpdf.(d_exp, tl)) : log1mexp.(logcdf.(d_exp, tu) + log1mexp.(logcdf.(d_exp, ceil.(tl) .- 1) - logcdf.(d_exp, tu)))
    return expert_tn_bar
end

## Parameters
params(d::GammaCountExpert) = (d.m, d.s)
function params_init(y, d::GammaCountExpert)
    
    # function init_obj(logparams, y)
    #     n = length(y)
    #     m_tmp = exp(logparams[1])
    #     s_tmp = exp(logparams[2])

    #     return -1 * ( sum(logpdf.(GammaCountExpert(m_tmp, s_tmp), y) ))
    # end

    # logparams_init = Optim.minimizer( Optim.optimize(x -> init_obj(x, y),  [log(2.0), 0.0] ))

    # m_init = exp(logparams_init[1])
    # s_init = exp(logparams_init[2])

    μ, σ2 = mean(y), var(y)
    s_init = σ2 / μ
    m_init = μ/(s_init)    

    try 
        GammaCountExpert(m_init, s_init) 
    catch; 
        GammaCountExpert() 
    end
end

## Simululation
sim_expert(d::GammaCountExpert, sample_size) = Distributions.rand(LRMoE.GammaCount(d.m, d.s), sample_size)

## penalty
penalty_init(d::GammaCountExpert) = [2.0 10.0 2.0 10.0]
no_penalty_init(d::GammaCountExpert) = [1.0 Inf 1.0 Inf]
penalize(d::GammaCountExpert, p) = (p[1]-1)*log(d.m) - d.m/p[2] + (p[3]-1)*log(d.s) - d.s/p[4]

## statistics
mean(d::GammaCountExpert) = mean(LRMoE.GammaCount(d.m, d.s))
var(d::GammaCountExpert) = var(LRMoE.GammaCount(d.m, d.s))
quantile(d::GammaCountExpert, p) = quantile(LRMoE.GammaCount(d.m, d.s), p)

## Misc functions for E-Step

function _sum_dens_series(m_new, s_new, d::GammaCountExpert, yl, yu)
    upper_finite = isinf(yu) ? Distributions.quantile(GammaCount(d.m, d.s), 1-1e-10) : yu
    series = yl:(max(yl, min(yu, upper_finite+1)))
    return sum(logpdf.(GammaCountExpert(m_new, s_new), series) .* pdf.(d, series))[1]
end

function _int_obs_dens_raw(m_new, s_new, d::GammaCountExpert, yl, yu)
    return _sum_dens_series(m_new, s_new, d, yl, yu)
end

function _int_lat_dens_raw(m_new, s_new, d::GammaCountExpert, tl, tu)
    return (tl==0 ? 0.0 : _sum_dens_series.(m_new, s_new, d, 0, ceil(tl)-1)) + (isinf(tu) ? 0.0 : _sum_dens_series.(m_new, s_new, d, floor(tu)+1, Inf))
end


function _gammacount_optim_params(lognew,
                        d_old,
                        tl, yl, yu, tu,
                        expert_ll_pos, expert_tn_pos, expert_tn_bar_pos,
                        z_e_obs, z_e_lat, k_e; # ,
                        # Y_e_obs, Y_e_lat;
                        penalty = true, pen_pararms_jk = [1.0 Inf 1.0 Inf])
    # Optimization in log scale for unconstrained computation    
    m_tmp = exp(lognew[1])
    s_tmp = exp(lognew[2])

    # Further E-Step
    yl_yu_unique = unique_bounds(yl, yu)
    int_obs_dens_tmp = _int_obs_dens_raw.(m_tmp, s_tmp, d_old, yl_yu_unique[:,1], yl_yu_unique[:,2])
    densY_e_obs = exp.(-expert_ll_pos) .* int_obs_dens_tmp[match_unique_bounds(hcat(vec(yl), vec(yu)), yl_yu_unique)]
    nan2num(densY_e_obs, 0.0) # get rid of NaN

    tl_tu_unique = unique_bounds(tl, tu)
    int_lat_dens_tmp = _int_lat_dens_raw.(m_tmp, s_tmp, d_old, tl_tu_unique[:,1], tl_tu_unique[:,2])
    densY_e_lat = exp.(-expert_tn_bar_pos) .* int_lat_dens_tmp[match_unique_bounds(hcat(vec(tl), vec(tu)), tl_tu_unique)]
    nan2num(densY_e_lat, 0.0) # get rid of NaN

    # term_zkz = z_e_obs .+ (z_e_lat .* k_e)
    term_zkz_Y = (z_e_obs .* densY_e_obs) .+ (z_e_lat .* k_e .* densY_e_lat)

    # sum_term_zkz = sum(term_zkz)[1]
    sum_term_zkzy = sum(term_zkz_Y)[1]

    obj = sum_term_zkzy
    p = penalty ? (pen_pararms_jk[1]-1)*log(m_tmp) - m_tmp/pen_pararms_jk[2] + (pen_pararms_jk[3]-1)*log(s_tmp) - s_tmp/pen_pararms_jk[4] : 0.0
    return (obj + p) * (-1.0)
end

## EM: M-Step
function EM_M_expert(d::GammaCountExpert,
                    tl, yl, yu, tu,
                    expert_ll_pos,
                    expert_tn_pos,
                    expert_tn_bar_pos,
                    z_e_obs, z_e_lat, k_e;
                    penalty = true, pen_pararms_jk = [1.0 Inf 1.0 Inf])

    # Update parameters
    logparams_new = Optim.minimizer( Optim.optimize(x -> _gammacount_optim_params(x, d,
                                                tl, yl, yu, tu,
                                                expert_ll_pos, expert_tn_pos, expert_tn_bar_pos,
                                                z_e_obs, z_e_lat, k_e,
                                                # Y_e_obs, Y_e_lat,
                                                penalty = penalty, pen_pararms_jk = pen_pararms_jk),
                                                # [log(d.m)-2.0, log(d.s)-2.0],
                                                # [log(d.m)+2.0, log(d.s)+2.0],
                                                [log(d.m), log(d.s)] ))
    # println("$logparams_new")
    m_new = exp(logparams_new[1])
    s_new = exp(logparams_new[2])

    # println("$m_new, $s_new")

    # Deal with zero mass
    if ccdf.(Gamma((0+1)*s_new, 1), m_new*s_new) > 0.999999 || isnan(ccdf.(Gamma((0+1)*s_new, 1), m_new*s_new))
        m_new, s_new = d.m, d.s
    end

    return GammaCountExpert(m_new, s_new)

end

## EM: M-Step, exact observations
function _gammacount_optim_params(lognew,
                        d_old,
                        ye, # tl, yl, yu, tu,
                        expert_ll_pos, # expert_tn_pos, expert_tn_bar_pos,
                        z_e_obs; # , # z_e_lat, k_e,
                        # Y_e_obs, Y_e_lat;
                        penalty = true, pen_pararms_jk = [1.0 Inf 1.0 Inf])
    # Optimization in log scale for unconstrained computation    
    m_tmp = exp(lognew[1])
    s_tmp = exp(lognew[2])

    # Further E-Step
    # yl_yu_unique = unique_bounds(yl, yu)
    # int_obs_dens_tmp = _int_obs_dens_raw.(m_tmp, s_tmp, d_old, yl_yu_unique[:,1], yl_yu_unique[:,2])
    densY_e_obs = logpdf.(GammaCountExpert(m_tmp, s_tmp), ye) # exp.(-expert_ll_pos) .* int_obs_dens_tmp[match_unique_bounds(hcat(vec(yl), vec(yu)), yl_yu_unique)]
    nan2num(densY_e_obs, 0.0) # get rid of NaN

    # tl_tu_unique = unique_bounds(tl, tu)
    # int_lat_dens_tmp = _int_lat_dens_raw.(m_tmp, s_tmp, d_old, tl_tu_unique[:,1], tl_tu_unique[:,2])
    densY_e_lat = 0.0 # exp.(-expert_tn_bar_pos) .* int_lat_dens_tmp[match_unique_bounds(hcat(vec(tl), vec(tu)), tl_tu_unique)]
    # nan2num(densY_e_lat, 0.0) # get rid of NaN

    # term_zkz = z_e_obs .+ (z_e_lat .* k_e)
    term_zkz_Y = (z_e_obs .* densY_e_obs) # .+ (z_e_lat .* k_e .* densY_e_lat)

    # sum_term_zkz = sum(term_zkz)[1]
    sum_term_zkzy = sum(term_zkz_Y)[1]

    obj = sum_term_zkzy
    p = penalty ? (pen_pararms_jk[1]-1)*log(m_tmp) - m_tmp/pen_pararms_jk[2] + (pen_pararms_jk[3]-1)*log(s_tmp) - s_tmp/pen_pararms_jk[4] : 0.0
    return (obj + p) * (-1.0)
end
function EM_M_expert_exact(d::GammaCountExpert,
                    ye,
                    expert_ll_pos,
                    z_e_obs; 
                    penalty = true, pen_pararms_jk = [1.0 Inf 1.0 Inf])

    # Update parameters
    logparams_new = Optim.minimizer( Optim.optimize(x -> _gammacount_optim_params(x, d,
                                                ye, # tl, yl, yu, tu,
                                                expert_ll_pos, # expert_tn_pos, expert_tn_bar_pos,
                                                z_e_obs, # z_e_lat, k_e,
                                                # Y_e_obs, Y_e_lat,
                                                penalty = penalty, pen_pararms_jk = pen_pararms_jk),
                                                # [log(d.m)-2.0, log(d.s)-2.0],
                                                # [log(d.m)+2.0, log(d.s)+2.0],
                                                [log(d.m), log(d.s)] ))

    # println("$logparams_new")
    m_new = exp(logparams_new[1])
    s_new = exp(logparams_new[2])

    # Deal with zero mass
    if ccdf.(Gamma((0+1)*s_new, 1), m_new*s_new) > 0.999999 || isnan(ccdf.(Gamma((0+1)*s_new, 1), m_new*s_new))
        m_new, s_new = d.m, d.s
    end
    
    # println("$m_new, $s_new")
    return GammaCountExpert(m_new, s_new)

end