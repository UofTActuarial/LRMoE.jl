"""
    ZILogNormalExpert(p, μ, σ)

Expert function: `ZILogNormalExpert(p, μ, σ)`.

"""
struct ZILogNormalExpert{T<:Real} <: ZIContinuousExpert
    p::T
    μ::T
    σ::T
    ZILogNormalExpert{T}(p::T, µ::T, σ::T) where {T<:Real} = new{T}(p, µ, σ)
end


function ZILogNormalExpert(p::T, μ::T, σ::T; check_args=true) where {T <: Real}
    check_args && @check_args(ZILogNormalExpert, 0 <= p <= 1 && σ >= zero(σ))
    return ZILogNormalExpert{T}(p, μ, σ)
end


#### Outer constructors
ZILogNormalExpert(p::Real, μ::Real, σ::Real) = ZILogNormalExpert(promote(p, μ, σ)...)
ZILogNormalExpert(p::Integer, μ::Integer, σ::Integer) = ZILogNormalExpert(float(p), float(μ), float(σ))

## Conversion
function convert(::Type{ZILogNormalExpert{T}}, p::S, μ::S, σ::S) where {T <: Real, S <: Real}
    ZILogNormalExpert(T(p), T(μ), T(σ))
end
function convert(::Type{ZILogNormalExpert{T}}, d::ZILogNormalExpert{S}) where {T <: Real, S <: Real}
    ZILogNormalExpert(T(d.p), T(d.μ), T(d.σ), check_args=false)
end
copy(d::ZILogNormalExpert) = ZILogNormalExpert(d.p, d.μ, d.σ, check_args=false)

## Loglikelihood of Expoert
logpdf(d::ZILogNormalExpert, x...) = Distributions.logpdf.(Distributions.LogNormal(d.μ, d.σ), x...)
pdf(d::ZILogNormalExpert, x...) = Distributions.pdf.(Distributions.LogNormal(d.μ, d.σ), x...)
logcdf(d::ZILogNormalExpert, x...) = Distributions.logcdf.(Distributions.LogNormal(d.μ, d.σ), x...)
cdf(d::ZILogNormalExpert, x...) = Distributions.cdf.(Distributions.LogNormal(d.μ, d.σ), x...)