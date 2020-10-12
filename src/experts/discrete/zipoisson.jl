"""
    ZIPoissonExpert(λ)

Expert function: `ZIPoissonExpert(λ)`.

"""
struct ZIPoissonExpert{T<:Real} <: ZIDiscreteExpert
    p::T
    λ::T
    ZIPoissonExpert{T}(p::T, λ::T) where {T<:Real} = new{T}(p, λ)
end

function ZIPoissonExpert(p::T, λ::T; check_args=true) where {T <: Real}
    check_args && @check_args(ZIPoissonExpert, 0 <= p <= 1 && λ >= zero(λ))
    return ZIPoissonExpert{T}(p, λ)
end

## Outer constructors
ZIPoissonExpert(p::Real, λ::Real) = ZIPoissonExpert(promote(p, λ)...)
ZIPoissonExpert(p::Integer, λ::Integer) = ZIPoissonExpert(float(p), float(λ))

## Loglikelihood of Expoert
logpdf(d::ZIPoissonExpert, x...) = isinf(x...) ? 0.0 : Distributions.logpdf.(Distributions.Poisson(d.λ), x...)
pdf(d::ZIPoissonExpert, x...) = isinf(x...) ? -Inf : Distributions.pdf.(Distributions.Poisson(d.λ), x...)
logcdf(d::ZIPoissonExpert, x...) = isinf(x...) ? 0.0 : Distributions.logcdf.(Distributions.Poisson(d.λ), x...)
cdf(d::ZIPoissonExpert, x...) = isinf(x...) ? 1.0 : Distributions.cdf.(Distributions.Poisson(d.λ), x...)