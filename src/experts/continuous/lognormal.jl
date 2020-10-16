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
    LogNormalExpert(T(d.μ), T(d.λ), check_args=false)
end
copy(d::LogNormalExpert) = LogNormalExpert(d.μ, d.σ, check_args=false)

## Loglikelihood of Expoert
logpdf(d::LogNormalExpert, x...) = Distributions.logpdf.(Distributions.LogNormal(d.μ, d.σ), x...)
pdf(d::LogNormalExpert, x...) = Distributions.pdf.(Distributions.LogNormal(d.μ, d.σ), x...)
logcdf(d::LogNormalExpert, x...) = Distributions.logcdf.(Distributions.LogNormal(d.μ, d.σ), x...)
cdf(d::LogNormalExpert, x...) = Distributions.cdf.(Distributions.LogNormal(d.μ, d.σ), x...)