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

function GammaCountExpert(m::T, s::T; check_args=true) where {T<:Real}
    check_args && @check_args(GammaCountExpert, m > zero(m) && s > zero(s))
    return GammaCountExpert{T}(m, s)
end

## Outer constructors
GammaCountExpert(m::Real, s::Real) = GammaCountExpert(promote(m, s)...)
GammaCountExpert(m::Integer, s::Integer) = GammaCountExpert(float(m), float(s))
GammaCountExpert() = GammaCountExpert(2.0, 1.0)

## Conversion
function convert(::Type{GammaCountExpert{T}}, m::S, s::S) where {T<:Real,S<:Real}
    return GammaCountExpert(T(m), T(s))
end
function convert(
    ::Type{GammaCountExpert{T}}, d::GammaCountExpert{S}
) where {T<:Real,S<:Real}
    return GammaCountExpert(T(d.m), T(d.s); check_args=false)
end
copy(d::GammaCountExpert) = GammaCountExpert(d.m, d.s; check_args=false)

## Loglikelihood of Expoert
function logpdf(d::GammaCountExpert, x...)
    return isinf(x...) ? -Inf : LRMoE.logpdf.(LRMoE.GammaCount(d.m, d.s), x...)
end
function pdf(d::GammaCountExpert, x...)
    return isinf(x...) ? 0.0 : LRMoE.pdf.(LRMoE.GammaCount(d.m, d.s), x...)
end
function logcdf(d::GammaCountExpert, x...)
    return isinf(x...) ? 0.0 : LRMoE.logcdf.(LRMoE.GammaCount(d.m, d.s), x...)
end
function cdf(d::GammaCountExpert, x...)
    return isinf(x...) ? 1.0 : LRMoE.cdf.(LRMoE.GammaCount(d.m, d.s), x...)
end

## expert_ll, etc
expert_ll_exact(d::GammaCountExpert, x::Real) = LRMoE.logpdf(LRMoE.GammaCount(d.m, d.s), x)
function expert_ll(d::GammaCountExpert, tl::Real, yl::Real, yu::Real, tu::Real)
    # d_exp = LRMoE.GammaCount(d.m, d.s)
    expert_ll = if (yl == yu)
        logpdf.(d, yl)
    else
        logcdf.(d, yu) + log1mexp.(logcdf.(d, ceil.(yl) .- 1) - logcdf.(d, yu))
    end
    return expert_ll
end
function expert_tn(d::GammaCountExpert, tl::Real, yl::Real, yu::Real, tu::Real)
    # d_exp = LRMoE.GammaCount(d.m, d.s)
    expert_tn = if (tl == tu)
        logpdf.(d, tl)
    else
        logcdf.(d, tu) + log1mexp.(logcdf.(d, ceil.(tl) .- 1) - logcdf.(d, tu))
    end
    return expert_tn
end
function expert_tn_bar(d::GammaCountExpert, tl::Real, yl::Real, yu::Real, tu::Real)
    # d_exp = LRMoE.GammaCount(d.m, d.s)
    expert_tn_bar = if (tl == tu)
        log1mexp.(logpdf.(d, tl))
    else
        log1mexp.(logcdf.(d, tu) + log1mexp.(logcdf.(d, ceil.(tl) .- 1) - logcdf.(d, tu)))
    end
    return expert_tn_bar
end

exposurize_expert(d::GammaCountExpert; exposure=1) = GammaCountExpert(d.m, d.s / exposure)

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
    m_init = μ / (s_init)

    try
        GammaCountExpert(m_init, s_init)
    catch
        GammaCountExpert()
    end
end

## Simululation
sim_expert(d::GammaCountExpert) = Distributions.rand(LRMoE.GammaCount(d.m, d.s), 1)[1]

## penalty
penalty_init(d::GammaCountExpert) = [2.0 10.0 2.0 10.0]
no_penalty_init(d::GammaCountExpert) = [1.0 Inf 1.0 Inf]
function penalize(d::GammaCountExpert, p)
    return (p[1] - 1) * log(d.m) - d.m / p[2] + (p[3] - 1) * log(d.s) - d.s / p[4]
end

## statistics
mean(d::GammaCountExpert) = mean(LRMoE.GammaCount(d.m, d.s))
var(d::GammaCountExpert) = var(LRMoE.GammaCount(d.m, d.s))
quantile(d::GammaCountExpert, p) = quantile(LRMoE.GammaCount(d.m, d.s), p)

## Misc functions for E-Step

function _sum_dens_series(m_new, s_new, d::GammaCountExpert, yl, yu, exposure)
    upper_finite = isinf(yu) ? quantile(d, 1 - 1e-8) : yu
    series = yl:(max(yl, min(yu, upper_finite + 1)))
    return sum(
        logpdf.(GammaCountExpert(m_new, s_new / exposure), series) .* pdf.(d, series)
    )[1]
end

function _int_obs_dens_raw(m_new, s_new, d::GammaCountExpert, yl, yu, exposure)
    return _sum_dens_series(m_new, s_new, d, yl, yu, exposure)
end

function _int_lat_dens_raw(m_new, s_new, d::GammaCountExpert, tl, tu, exposure)
    return (tl == 0 ? 0.0 : _sum_dens_series.(m_new, s_new, d, 0, ceil(tl) - 1), exposure) +
           (
        isinf(tu) ? 0.0 : _sum_dens_series.(m_new, s_new, d, floor(tu) + 1, Inf, exposure)
    )
end

function _gammacount_optim_params(lognew,
    d_old,
    tl, yl, yu, tu,
    # expert_ll_pos, expert_tn_pos, expert_tn_bar_pos,
    exposure,
    z_e_obs, z_e_lat, k_e; # ,
    # Y_e_obs, Y_e_lat;
    penalty=true, pen_pararms_jk=[1.0 Inf 1.0 Inf])
    # Optimization in log scale for unconstrained computation    
    m_tmp = exp(lognew[1])
    s_tmp = exp(lognew[2])

    # Further E-Step
    # yl_yu_unique = unique_bounds(yl, yu)
    # int_obs_dens_tmp = _int_obs_dens_raw.(m_tmp, s_tmp, d_old, yl_yu_unique[:,1], yl_yu_unique[:,2])
    # densY_e_obs = exp.(-expert_ll_pos) .* int_obs_dens_tmp[match_unique_bounds(hcat(vec(yl), vec(yu)), yl_yu_unique)]
    # nan2num(densY_e_obs, 0.0) # get rid of NaN

    # tl_tu_unique = unique_bounds(tl, tu)
    # int_lat_dens_tmp = _int_lat_dens_raw.(m_tmp, s_tmp, d_old, tl_tu_unique[:,1], tl_tu_unique[:,2])
    # densY_e_lat = exp.(-expert_tn_bar_pos) .* int_lat_dens_tmp[match_unique_bounds(hcat(vec(tl), vec(tu)), tl_tu_unique)]
    # nan2num(densY_e_lat, 0.0) # get rid of NaN

    densY_e_obs = fill(0.0, length(yl))
    densY_e_lat = fill(0.0, length(yl))
    d_tmp = LRMoE.GammaCountExpert(m_tmp, s_tmp)

    for i in 1:length(yl)
        d_expo = exposurize_expert(d_tmp; exposure=exposure[i])
        # expert_ll_pos = expert_ll(d_expo, tl[i], yl[i], yu[i], tu[i])
        # expert_tn_bar_pos = expert_tn_bar(d_expo, tl[i], yl[i], yu[i], tu[i])
        densY_e_obs[i] = expert_ll(d_expo, tl[i], yl[i], yu[i], tu[i])
        # _int_obs_dens_raw(m_tmp, s_tmp, d_expo, yl[i], yu[i], exposure[i]) # exp(-expert_ll_pos) * _int_obs_dens_raw(m_tmp, s_tmp, d_expo, yl[i], yu[i], exposure[i])
        densY_e_lat[i] = expert_tn_bar(d_expo, tl[i], yl[i], yu[i], tu[i])
        # _int_lat_dens_raw(m_tmp, s_tmp, d_expo, yl[i], yu[i], exposure[i]) # exp(-expert_tn_bar_pos) * _int_lat_dens_raw(m_tmp, s_tmp, d_expo, yl[i], yu[i], exposure[i])
    end

    nan2num(densY_e_obs, 0.0) # get rid of NaN
    nan2num(densY_e_lat, 0.0) # get rid of NaN
    inf2num(densY_e_obs, 0.0) # get rid of inf
    inf2num(densY_e_lat, 0.0) # get rid of inf

    # term_zkz = z_e_obs .+ (z_e_lat .* k_e)
    term_zkz_Y = (z_e_obs .* densY_e_obs) .+ (z_e_lat .* k_e .* densY_e_lat)

    # sum_term_zkz = sum(term_zkz)[1]
    sum_term_zkzy = sum(term_zkz_Y)[1]

    obj = sum_term_zkzy
    p = if penalty
        (pen_pararms_jk[1] - 1) * log(m_tmp) - m_tmp / pen_pararms_jk[2] +
        (pen_pararms_jk[3] - 1) * log(s_tmp) - s_tmp / pen_pararms_jk[4]
    else
        0.0
    end
    return (obj + p) * (-1.0)
end

## EM: M-Step
function EM_M_expert(d::GammaCountExpert,
    tl, yl, yu, tu,
    exposure,
    z_e_obs, z_e_lat, k_e;
    penalty=true, pen_pararms_jk=[1.0 Inf 1.0 Inf])

    # Update parameters
    logparams_new = Optim.minimizer(
        Optim.optimize(
            x -> _gammacount_optim_params(x, d,
                tl, yl, yu, tu,
                exposure,
                z_e_obs, z_e_lat, k_e;
                # Y_e_obs, Y_e_lat,
                penalty=penalty, pen_pararms_jk=pen_pararms_jk),
            # [log(d.m)-2.0, log(d.s)-2.0],
            # [log(d.m)+2.0, log(d.s)+2.0],
            [log(d.m), log(d.s)]),
    )
    # println("$logparams_new")
    m_new = exp(logparams_new[1])
    s_new = exp(logparams_new[2])

    # println("$m_new, $s_new")

    # Deal with zero mass
    if ccdf.(Gamma((0 + 1) * s_new, 1), m_new * s_new) > 0.999999 ||
        isnan(ccdf.(Gamma((0 + 1) * s_new, 1), m_new * s_new))
        m_new, s_new = d.m, d.s
    end

    return GammaCountExpert(m_new, s_new)
end

## EM: M-Step, exact observations
function _gammacount_optim_params(lognew,
    ye, exposure,
    z_e_obs;
    penalty=true, pen_pararms_jk=[1.0 Inf 1.0 Inf])
    # Optimization in log scale for unconstrained computation    
    m_tmp = exp(lognew[1])
    s_tmp = exp(lognew[2])

    # Further E-Step
    densY_e_obs = fill(NaN, length(exposure))
    for i in 1:length(exposure)
        densY_e_obs[i] = expert_ll_exact(
            exposurize_expert(LRMoE.GammaCountExpert(m_tmp, s_tmp); exposure=exposure[i]),
            ye[i],
        )
    end
    # densY_e_obs = logpdf.(GammaCountExpert(m_tmp, s_tmp), ye)
    nan2num(densY_e_obs, 0.0) # get rid of NaN

    term_zkz_Y = z_e_obs .* densY_e_obs

    sum_term_zkzy = sum(term_zkz_Y)[1]

    obj = sum_term_zkzy
    p = if penalty
        (pen_pararms_jk[1] - 1) * log(m_tmp) - m_tmp / pen_pararms_jk[2] +
        (pen_pararms_jk[3] - 1) * log(s_tmp) - s_tmp / pen_pararms_jk[4]
    else
        0.0
    end
    return (obj + p) * (-1.0)
end

function EM_M_expert_exact(d::GammaCountExpert,
    ye, exposure,
    z_e_obs;
    penalty=true, pen_pararms_jk=[1.0 Inf 1.0 Inf])

    # Update parameters
    logparams_new = Optim.minimizer(
        Optim.optimize(
            x -> _gammacount_optim_params(x,
                ye, exposure,
                z_e_obs;
                penalty=penalty, pen_pararms_jk=pen_pararms_jk),
            [log(d.m), log(d.s)]),
    )

    # println("$logparams_new")
    m_new = exp(logparams_new[1])
    s_new = exp(logparams_new[2])

    # Deal with zero mass
    if ccdf.(Gamma((0 + 1) * s_new, 1), m_new * s_new) > 0.999999 ||
        isnan(ccdf.(Gamma((0 + 1) * s_new, 1), m_new * s_new))
        m_new, s_new = d.m, d.s
    end

    # println("$m_new, $s_new")
    return GammaCountExpert(m_new, s_new)
end