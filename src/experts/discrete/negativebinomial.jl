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

function NegativeBinomialExpert(n::T, p::T; check_args=true) where {T<:Real}
    check_args && @check_args(NegativeBinomialExpert, 0 <= p <= 1 && n > zero(n))
    return NegativeBinomialExpert{T}(n, p)
end

## Outer constructors
NegativeBinomialExpert(n::Real, p::Real) = NegativeBinomialExpert(promote(n, p)...)
NegativeBinomialExpert(n::Integer, p::Integer) = NegativeBinomialExpert(float(n), float(p))
NegativeBinomialExpert() = NegativeBinomialExpert(1, 0.5)

## Conversion
function convert(::Type{NegativeBinomialExpert{T}}, n::S, p::S) where {T<:Real,S<:Real}
    return NegativeBinomialExpert(T(n), T(p))
end
function convert(
    ::Type{NegativeBinomialExpert{T}}, d::NegativeBinomialExpert{S}
) where {T<:Real,S<:Real}
    return NegativeBinomialExpert(T(d.n), T(d.p); check_args=false)
end
copy(d::NegativeBinomialExpert) = NegativeBinomialExpert(d.n, d.p; check_args=false)

## Loglikelihood of Expoert
function logpdf(d::NegativeBinomialExpert, x...)
    return if isinf(x...)
        -Inf
    else
        Distributions.logpdf.(Distributions.NegativeBinomial(d.n, d.p), x...)
    end
end
function pdf(d::NegativeBinomialExpert, x...)
    return if isinf(x...)
        0.0
    else
        Distributions.pdf.(Distributions.NegativeBinomial(d.n, d.p), x...)
    end
end
function logcdf(d::NegativeBinomialExpert, x...)
    return if isinf(x...)
        0.0
    else
        Distributions.logcdf.(Distributions.NegativeBinomial(d.n, d.p), x...)
    end
end
function cdf(d::NegativeBinomialExpert, x...)
    return if isinf(x...)
        1.0
    else
        Distributions.cdf.(Distributions.NegativeBinomial(d.n, d.p), x...)
    end
end

## expert_ll, etc
function expert_ll_exact(d::NegativeBinomialExpert, x::Real)
    return LRMoE.logpdf(LRMoE.NegativeBinomialExpert(d.n, d.p), x)
end
function expert_ll(d::NegativeBinomialExpert, tl::Real, yl::Real, yu::Real, tu::Real)
    # d_exp = NegativeBinomialExpert(d.n*exposure, d.p)
    expert_ll = if (yl == yu)
        logpdf.(d, yl)
    else
        logcdf.(d, yu) + log1mexp.(logcdf.(d, ceil.(yl) .- 1) - logcdf.(d, yu))
    end
    return expert_ll
end
function expert_tn(d::NegativeBinomialExpert, tl::Real, yl::Real, yu::Real, tu::Real)
    # d_exp = NegativeBinomialExpert(d.n*exposure, d.p)
    expert_tn = if (tl == tu)
        logpdf.(d, tl)
    else
        logcdf.(d, tu) + log1mexp.(logcdf.(d, ceil.(tl) .- 1) - logcdf.(d, tu))
    end
    return expert_tn
end
function expert_tn_bar(d::NegativeBinomialExpert, tl::Real, yl::Real, yu::Real, tu::Real)
    # d_exp = NegativeBinomialExpert(d.n*exposure, d.p)
    expert_tn_bar = if (tl == tu)
        log1mexp.(logpdf.(d, tl))
    else
        log1mexp.(logcdf.(d, tu) + log1mexp.(logcdf.(d, ceil.(tl) .- 1) - logcdf.(d, tu)))
    end
    return expert_tn_bar
end

function exposurize_expert(d::NegativeBinomialExpert; exposure=1)
    return NegativeBinomialExpert(d.n * exposure, d.p)
end

## Parameters
params(d::NegativeBinomialExpert) = (d.n, d.p)
function params_init(y, d::NegativeBinomialExpert)
    μ, σ2 = mean(y), var(y)
    p_init = μ / σ2
    n_init = μ * p_init / (1 - p_init)
    try
        NegativeBinomialExpert(n_init, p_init)
    catch
        NegativeBinomialExpert()
    end
end

## Simululation
function sim_expert(d::NegativeBinomialExpert)
    return Distributions.rand(Distributions.NegativeBinomial(d.n, d.p), 1)[1]
end

## penalty
penalty_init(d::NegativeBinomialExpert) = [2.0 10.0]
no_penalty_init(d::NegativeBinomialExpert) = [1.0 Inf]
penalize(d::NegativeBinomialExpert, p) = (p[1] - 1) * log(d.n) - d.n / p[2]

## statistics
mean(d::NegativeBinomialExpert) = mean(Distributions.NegativeBinomial(d.n, d.p))
var(d::NegativeBinomialExpert) = var(Distributions.NegativeBinomial(d.n, d.p))
function quantile(d::NegativeBinomialExpert, p)
    return quantile(Distributions.NegativeBinomial(d.n, d.p), p)
end

## Misc functions for E-Step

function _sum_densy_series(d::NegativeBinomialExpert, yl, yu)
    if isinf(yu)
        series = 0:(max(yl - 1, 0))
        return d.n * (1 - d.p) / d.p - sum(pdf.(d, series) .* series)[1]
    else
        series = yl:yu
        return sum(pdf.(d, series) .* series)[1]
    end
end

function _int_obs_Y_raw(d::NegativeBinomialExpert, yl, yu)
    return _sum_densy_series(d, yl, yu)
end

function _int_lat_Y_raw(d::NegativeBinomialExpert, tl, tu)
    return d.n * (1 - d.p) / d.p - _sum_densy_series(d, tl, tu)
end

function _sum_denslogy_series(n_new, d::NegativeBinomialExpert, yl, yu, exposure)
    upper_finite = isinf(yu) ? quantile(d, 1 - 1e-8) : yu
    series = yl:(max(yl, min(yu, upper_finite + 1)))
    # series = yl:yu
    return sum(pdf.(d, series) .* loggamma.(series .+ n_new * exposure))[1]
end

function _int_obs_logY_raw(n_new, d::NegativeBinomialExpert, yl, yu, exposure)
    if yl == yu
        return pdf.(d, yl) .* loggamma.(yl .+ n_new * exposure)
    else
        return _sum_denslogy_series.(n_new, d, yl, yu, exposure)
    end
end

function _int_lat_logY_raw(n_new, d::NegativeBinomialExpert, tl, tu, exposure)
    # return (tl==0 ? 0.0 : _sum_denslogy_series.(n_new, d, 0, tl)) + (isinf(tu) ? 0.0 : _sum_denslogy_series.(n_new, d, tu, Inf))
    return (tl == 0 ? 0.0 : _sum_denslogy_series.(n_new, d, 0, ceil(tl) - 1, exposure)) +
           (isinf(tu) ? 0.0 : _sum_denslogy_series.(n_new, d, floor(tu) + 1, Inf, exposure))
end

function _negativebinomial_n_to_p(n, sum_term_zkz, sum_term_zkzy;
    penalty=true, pen_pararms_jk=[])
    return 1 - sum_term_zkzy / (sum_term_zkz + sum_term_zkzy)
end

function _negativebinomial_optim_n(logn,
    d_old,
    tl, yl, yu, tu,
    exposure,
    z_e_obs, z_e_lat, k_e,
    Y_e_obs, Y_e_lat;
    penalty=true, pen_pararms_jk=[])
    # Optimization in log scale for unconstrained computation    
    n_tmp = exp(logn)

    # Further E-Step
    # yl_yu_unique = unique_bounds(yl, yu)
    # int_obs_logY_tmp = _int_obs_logY_raw.(n_tmp, d_old, yl_yu_unique[:,1], yl_yu_unique[:,2])
    # logY_e_obs = exp.(-expert_ll_pos) .* int_obs_logY_tmp[match_unique_bounds(hcat(vec(yl), vec(yu)), yl_yu_unique)]
    # nan2num(logY_e_obs, 0.0) # get rid of NaN

    # tl_tu_unique = unique_bounds(tl, tu)
    # int_lat_logY_tmp = _int_lat_logY_raw.(n_tmp, d_old, tl_tu_unique[:,1], tl_tu_unique[:,2])
    # logY_e_lat = exp.(-expert_tn_bar_pos) .* int_lat_logY_tmp[match_unique_bounds(hcat(vec(tl), vec(tu)), tl_tu_unique)]
    # nan2num(logY_e_lat, 0.0) # get rid of NaN

    logY_e_obs = fill(0.0, length(yl))
    logY_e_lat = fill(0.0, length(yl))

    for i in 1:length(yl)
        d_expo = exposurize_expert(d_old; exposure=exposure[i])
        expert_ll_pos = expert_ll.(d_expo, tl[i], yl[i], yu[i], tu[i])
        expert_tn_bar_pos = expert_tn_bar.(d_expo, tl[i], yl[i], yu[i], tu[i])
        logY_e_obs[i] =
            exp(-expert_ll_pos) *
            _int_obs_logY_raw(n_tmp, d_expo, yl[i], yu[i], exposure[i])
        logY_e_lat[i] =
            exp(-expert_tn_bar_pos) *
            _int_lat_logY_raw(n_tmp, d_expo, tl[i], tu[i], exposure[i])
    end

    nan2num(logY_e_obs, 0.0) # get rid of NaN
    nan2num(logY_e_lat, 0.0) # get rid of NaN
    inf2num(logY_e_obs, 0.0) # get rid of inf
    inf2num(logY_e_lat, 0.0) # get rid of inf

    term_zkz = (z_e_obs .* n_tmp .* exposure) .+ (z_e_lat .* k_e .* n_tmp .* exposure)
    term_zkz_Y = (z_e_obs .* Y_e_obs) .+ (z_e_lat .* k_e .* Y_e_lat)
    term_zkz_logY = (z_e_obs .* logY_e_obs) .+ (z_e_lat .* k_e .* logY_e_lat)

    sum_term_zkz = sum(term_zkz)[1]
    sum_term_zkzy = sum(term_zkz_Y)[1]
    sum_term_zkzlogy = sum(term_zkz_logY)[1]

    p_tmp = _negativebinomial_n_to_p(
        n_tmp, sum_term_zkz, sum_term_zkzy; penalty=penalty, pen_pararms_jk=pen_pararms_jk
    )

    obj =
        sum_term_zkzlogy - sum(z_e_obs .* loggamma.(n_tmp .* exposure))[1] +
        sum_term_zkz * log(p_tmp) + sum_term_zkzy * log(1 - p_tmp)
    p = penalty ? (pen_pararms_jk[1] - 1) * log(n_tmp) - n_tmp / pen_pararms_jk[2] : 0.0
    return (obj + p) * (-1.0)
end

## EM: M-Step
function EM_M_expert(d::NegativeBinomialExpert,
    tl, yl, yu, tu,
    exposure,
    z_e_obs, z_e_lat, k_e;
    penalty=true, pen_pararms_jk=[2.0 1.0])

    # Further E-Step
    # yl_yu_unique = unique_bounds(yl, yu)
    # int_obs_Y_tmp = _int_obs_Y_raw.(d, yl_yu_unique[:,1], yl_yu_unique[:,2])
    # Y_e_obs = exp.(-expert_ll_pos) .* int_obs_Y_tmp[match_unique_bounds(hcat(vec(yl), vec(yu)), yl_yu_unique)]
    # nan2num(Y_e_obs, 0.0) # get rid of NaN

    # tl_tu_unique = unique_bounds(tl, tu)
    # int_lat_Y_tmp = _int_lat_Y_raw.(d, tl_tu_unique[:,1], tl_tu_unique[:,2])
    # Y_e_lat = exp.(-expert_tn_bar_pos) .* int_lat_Y_tmp[match_unique_bounds(hcat(vec(tl), vec(tu)), tl_tu_unique)]
    # nan2num(Y_e_lat, 0.0) # get rid of NaN
    Y_e_obs = fill(0.0, length(yl))
    Y_e_lat = fill(0.0, length(yl))

    for i in 1:length(yl)
        d_expo = exposurize_expert(d; exposure=exposure[i])
        expert_ll_pos = expert_ll.(d_expo, tl[i], yl[i], yu[i], tu[i])
        expert_tn_bar_pos = expert_tn_bar.(d_expo, tl[i], yl[i], yu[i], tu[i])
        Y_e_obs[i] = exp(-expert_ll_pos) * _int_obs_Y_raw(d_expo, yl[i], yu[i])
        Y_e_lat[i] = exp(-expert_tn_bar_pos) * _int_lat_Y_raw(d_expo, tl[i], tu[i])
    end

    nan2num(Y_e_obs, 0.0) # get rid of NaN
    nan2num(Y_e_lat, 0.0) # get rid of NaN
    inf2num(Y_e_obs, 0.0) # get rid of inf
    inf2num(Y_e_lat, 0.0) # get rid of inf

    # Update parameters
    logn_new = Optim.minimizer(
        Optim.optimize(
            x -> _negativebinomial_optim_n(x, d,
                tl, yl, yu, tu,
                exposure,
                z_e_obs, z_e_lat, k_e,
                Y_e_obs, Y_e_lat;
                penalty=penalty, pen_pararms_jk=pen_pararms_jk),
            log(d.n) - 0.5, log(d.n) + 0.5),
    ) # ,
    # max(log(d.n)-2.0, 0.0), log(d.n)+2.0 )) # ,
    # log(d.k)-2.0, log(d.k)+2.0 )) # ,
    # GoldenSection() )) 
    # , rel_tol = 1e-8) )
    n_new = exp(logn_new)

    term_zkz = (z_e_obs .* n_new .* exposure) .+ (z_e_lat .* k_e .* n_new .* exposure)
    term_zkz_Y = (z_e_obs .* Y_e_obs) .+ (z_e_lat .* k_e .* Y_e_lat)

    sum_term_zkz = sum(term_zkz)[1]
    sum_term_zkzy = sum(term_zkz_Y)[1]

    p_new = _negativebinomial_n_to_p(
        n_new, sum_term_zkz, sum_term_zkzy; penalty=penalty, pen_pararms_jk=pen_pararms_jk
    )

    # Deal with zero mass 
    if (p_new^n_new > 0.999999) || (isnan(p_new^n_new))
        p_new, n_new = d.p, d.n
    end

    return NegativeBinomialExpert(n_new, p_new)
end

## EM: M-Step, exact observations
function _negativebinomial_optim_n_exact(logn, d_old,
    ye, exposure,
    z_e_obs;
    penalty=true, pen_pararms_jk=[])
    # Optimization in log scale for unconstrained computation    
    n_tmp = exp(logn)

    Y_e_obs = ye

    # Further E-Step
    logY_e_obs = loggamma.(ye .+ n_tmp .* exposure)
    nan2num(logY_e_obs, 0.0) # get rid of NaN

    term_zkz = z_e_obs .* n_tmp .* exposure
    term_zkz_Y = z_e_obs .* Y_e_obs
    term_zkz_logY = z_e_obs .* logY_e_obs

    sum_term_zkz = sum(term_zkz)[1]
    sum_term_zkzy = sum(term_zkz_Y)[1]
    sum_term_zkzlogy = sum(term_zkz_logY)[1]

    p_tmp = _negativebinomial_n_to_p(
        n_tmp, sum_term_zkz, sum_term_zkzy; penalty=penalty, pen_pararms_jk=pen_pararms_jk
    )

    obj =
        sum_term_zkzlogy - sum(z_e_obs .* loggamma.(n_tmp .* exposure))[1] +
        sum_term_zkz * log(p_tmp) + sum_term_zkzy * log(1 - p_tmp)
    p = penalty ? (pen_pararms_jk[1] - 1) * log(n_tmp) - n_tmp / pen_pararms_jk[2] : 0.0
    return (obj + p) * (-1.0)
end

function EM_M_expert_exact(d::NegativeBinomialExpert,
    ye, exposure,
    # expert_ll_pos,
    z_e_obs;
    penalty=true, pen_pararms_jk=[1.0 Inf])

    # Update parameters
    logn_new = Optim.minimizer(
        Optim.optimize(
            x -> _negativebinomial_optim_n_exact(x, d,
                ye, exposure,
                z_e_obs;
                penalty=penalty, pen_pararms_jk=pen_pararms_jk),
            log(d.n) - 0.5, log(d.n) + 0.5),
    )
    n_new = exp(logn_new)

    term_zkz = z_e_obs .* n_new .* exposure
    term_zkz_Y = z_e_obs .* ye

    sum_term_zkz = sum(term_zkz)[1]
    sum_term_zkzy = sum(term_zkz_Y)[1]

    p_new = _negativebinomial_n_to_p(
        n_new, sum_term_zkz, sum_term_zkzy; penalty=penalty, pen_pararms_jk=pen_pararms_jk
    )

    # Deal with zero mass 
    if (p_new^n_new > 0.999999) || (isnan(p_new^n_new))
        p_new, n_new = d.p, d.n
    end

    return NegativeBinomialExpert(n_new, p_new)
end