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

## Loglikelihood of Expoert
logpdf(d::ZILogNormalExpert, x...) = Distributions.logpdf.(LogNormal(d.μ, d.σ), x...)
pdf(d::ZILogNormalExpert, x...) = Distributions.pdf.(LogNormal(d.μ, d.σ), x...)
logcdf(d::ZILogNormalExpert, x...) = Distributions.logcdf.(LogNormal(d.μ, d.σ), x...)
cdf(d::ZILogNormalExpert, x...) = Distributions.cdf.(LogNormal(d.μ, d.σ), x...)