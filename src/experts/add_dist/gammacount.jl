"""
    GammaCount(m, s)

"""
struct GammaCount{T<:Real} <: DiscreteUnivariateDistribution
    m::T
    s::T
    GammaCount{T}(m::Real, s::Real) where {T<:Real} = new{T}(m, s)
end

function GammaCount(m::T, s::T; check_args=true) where {T<:Real}
    check_args && @check_args(GammaCount, m > zero(m) && s > zero(s))
    return GammaCount{T}(m, s)
end

GammaCount(m::Real, s::Real) = GammaCount(promote(m, s)...)
GammaCount(m::Integer, s::Integer) = GammaCount(float(m), float(s))
# Poisson() = Poisson(1.0, check_args=false)

@distr_support GammaCount 0 Inf

#### Conversions
convert(::Type{GammaCount{T}}, m::S, s::S) where {T<:Real,S<:Real} = GammaCount(T(m), T(s))
function convert(::Type{GammaCount{T}}, d::GammaCount{S}) where {T<:Real,S<:Real}
    return GammaCount(T(d.m), T(d.s); check_args=false)
end

### Parameters

params(d::GammaCount) = (d.m, d.s)
partype(::GammaCount{T}) where {T} = T

# rate(d::Poisson) = d.λ

### Evaluation

# @_delegate_statsfuns Poisson pois λ

function pdf(d::GammaCount, x::T) where {T<:Real} # where {T <: Integer}
    if x < 0 || x != floor(x) || isinf(x)
        return 0.0
    elseif x == 0
        return ccdf.(Gamma((0 + 1) * d.s, 1), d.m * d.s) # 1-cdf.(Gamma((0+1)*d.s, 1), d.m*d.s)    
    else
        return cdf.(Gamma((floor(x)) * d.s, 1), d.m * d.s) -
               cdf.(Gamma((floor(x) + 1) * d.s, 1), d.m * d.s)
    end
end

pdf(d::GammaCount, x::T) where {T<:Integer} = pdf(d, convert(Float64, x))

function logpdf(d::GammaCount, x::T) where {T<:Real} # where {T <: Integer}
    if x < 0 || x != floor(x) || isinf(x)
        return -Inf
    elseif x == 0
        return logccdf.(Gamma((0 + 1) * d.s, 1), d.m * d.s)
    else
        return logcdf.(Gamma((floor(x)) * d.s, 1), d.m * d.s) +
               log1mexp.(
            logcdf.(Gamma((floor(x) + 1) * d.s, 1), d.m * d.s) -
            logcdf.(Gamma((floor(x)) * d.s, 1), d.m * d.s)
        )
    end
end

logpdf(d::GammaCount, x::T) where {T<:Integer} = logpdf(d, convert(Float64, x))

function cdf(d::GammaCount, x::T) where {T<:Real} # where {T <: Integer}
    if isinf(x)
        return 1.0
    elseif x < 0
        return 0.0
    else
        return ccdf.(Gamma((floor(x) + 1) * d.s, 1), d.m * d.s)
    end
end

cdf(d::GammaCount, x::T) where {T<:Integer} = cdf(d, convert(Float64, x))

function logcdf(d::GammaCount, x::T) where {T<:Real} # where {T <: Integer}
    if isinf(x)
        return 0.0
    elseif x < 0
        return -Inf
    else
        return log1mexp.(logcdf.(Gamma((floor(x) + 1) * d.s, 1), d.m * d.s))
    end
end

logcdf(d::GammaCount, x::T) where {T<:Integer} = logcdf(d, convert(Float64, x))

function quantile(d::GammaCount, q::Real)
    if q <= 0 || Distributions.cdf.(d, 0.0) >= q
        return 0
    elseif q >= 1
        return Inf
    else
        return _solve_discrete_quantile(d, q)
    end
end

function rand(rng::AbstractRNG, d::GammaCount)
    # u = 1 - rand(rng)
    return quantile(d, 1 - Base.rand(rng))
end

function _gc_moments(d::GammaCount, m)
    upper_finite = 250 # convert(Int, quantile(d, 1-1e-10))
    series = 0:upper_finite
    # return sum( (series.^m) .* LRMoE.pdf.(d, series) )[1]
    return sum((series .^ m) .* Distributions.pdf.(d, series))[1]
end

### Statistics

mean(d::GammaCount) = _gc_moments(d, 1)

# mode(d::GammaCount) = floor(Int,d.λ)

# function modes(d::Poisson)
#     λ = d.λ
#     isinteger(λ) ? [round(Int, λ) - 1, round(Int, λ)] : [floor(Int, λ)]
# end

var(d::GammaCount) = _gc_moments(d, 2) - (_gc_moments(d, 1))^2

function skewness(d::GammaCount)
    m1, m2, m3 = _gc_moments(d, 1), _gc_moments(d, 2), _gc_moments(d, 3)
    return (m3 - 3 * m1 * m2 - m1^3) / (m2^1.5)
end

function kurtosis(d::GammaCount)
    m1, m2, m3, m4 = _gc_moments(d, 1),
    _gc_moments(d, 2), _gc_moments(d, 3),
    _gc_moments(d, 4)
    return (m4 - 4 * m1 * m3 + 6 * (m1^2) * m2 - 3 * (m1)^4) / (m2^2)
end

# struct RecursiveGammaCountProbEvaluator <: RecursiveProbabilityEvaluator
#     m::Real
#     s::Real
# end

# RecursiveGammaCountProbEvaluator(d::GammaCount) = RecursiveGammaCountProbEvaluator(d.m, d.s)
# nextpdf(d::RecursiveGammaCountProbEvaluator, p::Float64, x::Integer) = p - ccdf.(Gamma((floor(x)+1)*d.s, 1), d.m*d.s) + ccdf.(Gamma((floor(x)+2)*d.s, 1), d.m*d.s)

# Base.broadcast!(::typeof(pdf), r::AbstractArray, d::GammaCount, rgn::UnitRange) =
#     _pdf!(r, d, rgn, RecursiveGammaCountProbEvaluator(d))
# function Base.broadcast(::typeof(pdf), d::GammaCount, X::UnitRange)
#     r = similar(Array{promote_type(partype(d), eltype(X))}, axes(X))
#     r .= pdf.(Ref(d),X)
# end

# function entropy(d::Poisson{T}) where T<:Real
#     λ = rate(d)
#     if λ == zero(T)
#         return zero(T)
#     elseif λ < 50
#         s = zero(T)
#         λk = one(T)
#         for k = 1:100
#             λk *= λ
#             s += λk * loggamma(k + 1) / gamma(k + 1)
#         end
#         return λ * (1 - log(λ)) + exp(-λ) * s
#     else
#         return log(2 * pi * ℯ * λ)/2 -
#                (1 / (12 * λ)) -
#                (1 / (24 * λ * λ)) -
#                (19 / (360 * λ * λ * λ))
#     end
# end

# function mgf(d::Poisson, t::Real)
#     λ = rate(d)
#     return exp(λ * (exp(t) - 1))
# end

# function cf(d::Poisson, t::Real)
#     λ = rate(d)
#     return exp(λ * (cis(t) - 1))
# end

# ### Fitting

# struct PoissonStats <: SufficientStats
#     sx::Float64   # (weighted) sum of x
#     tw::Float64   # total sample weight
# end

# suffstats(::Type{<:Poisson}, x::AbstractArray{T}) where {T<:Integer} = PoissonStats(sum(x), length(x))

# function suffstats(::Type{<:Poisson}, x::AbstractArray{T}, w::AbstractArray{Float64}) where T<:Integer
#     n = length(x)
#     n == length(w) || throw(DimensionMismatch("Inconsistent array lengths."))
#     sx = 0.
#     tw = 0.
#     for i = 1 : n
#         @inbounds wi = w[i]
#         @inbounds sx += x[i] * wi
#         tw += wi
#     end
#     PoissonStats(sx, tw)
# end

# fit_mle(::Type{<:Poisson}, ss::PoissonStats) = Poisson(ss.sx / ss.tw)

## samplers

# const poissonsampler_threshold = 6

# function sampler(d::Poisson)
#     if rate(d) < poissonsampler_threshold
#         return PoissonCountSampler(d)
#     else
#         return PoissonADSampler(d)
#     end
# end
