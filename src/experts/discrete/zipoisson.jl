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

## Conversion
function convert(::Type{ZIPoissonExpert{T}}, p::S, λ::S) where {T <: Real, S <: Real}
    ZIPoissonExpert(T(p), T(λ))
end
function convert(::Type{ZIPoissonExpert{T}}, d::ZIPoissonExpert{S}) where {T <: Real, S <: Real}
    ZIPoissonExpert(T(d.p), T(d.λ), check_args=false)
end
copy(d::ZIPoissonExpert) = ZIPoissonExpert(d.p, d.λ, check_args=false)

## Loglikelihood of Expoert
logpdf(d::ZIPoissonExpert, x...) = isinf(x...) ? 0.0 : Distributions.logpdf.(Distributions.Poisson(d.λ), x...)
pdf(d::ZIPoissonExpert, x...) = isinf(x...) ? -Inf : Distributions.pdf.(Distributions.Poisson(d.λ), x...)
logcdf(d::ZIPoissonExpert, x...) = isinf(x...) ? 0.0 : Distributions.logcdf.(Distributions.Poisson(d.λ), x...)
cdf(d::ZIPoissonExpert, x...) = isinf(x...) ? 1.0 : Distributions.cdf.(Distributions.Poisson(d.λ), x...)

## Simululation
sim_expert(d::ZIPoissonExpert, sample_size) = (1 .- Distributions.rand(Distributions.Bernoulli(d.p), sample_size)) .* Distributions.rand(Distributions.Poisson(d.λ), sample_size)

## penalty
penalty_init(d::ZIPoissonExpert) = [2.0 1.0]
penalize(d::ZIPoissonExpert, p) = (p[1]-1)*log(d.λ) - d.λ/p[2]