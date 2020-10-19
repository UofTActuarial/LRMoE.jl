"""
    PoissonExpert(λ)

Expert function: `PoissonExpert(λ)`.

"""
struct PoissonExpert{T<:Real} <: NonZIDiscreteExpert
    λ::T
    PoissonExpert{T}(λ::T) where {T<:Real} = new{T}(λ)
end

function PoissonExpert(λ::T; check_args=true) where {T <: Real}
    check_args && @check_args(PoissonExpert, λ >= zero(λ))
    return PoissonExpert{T}(λ)
end

## Outer constructors
# PoissonExpert(λ::Real) = PoissonExpert(promote(λ)...)
PoissonExpert(λ::Integer) = PoissonExpert(float(λ))

## Conversion
function convert(::Type{PoissonExpert{T}}, λ::S) where {T <: Real, S <: Real}
    PoissonExpert(T(λ))
end
function convert(::Type{PoissonExpert{T}}, d::PoissonExpert{S}) where {T <: Real, S <: Real}
    PoissonExpert(T(d.λ), check_args=false)
end
copy(d::PoissonExpert) = PoissonExpert(d.λ, check_args=false)

## Loglikelihood of Expoert
logpdf(d::PoissonExpert, x...) = isinf(x...) ? 0.0 : Distributions.logpdf.(Distributions.Poisson(d.λ), x...)
pdf(d::PoissonExpert, x...) = isinf(x...) ? -Inf : Distributions.pdf.(Distributions.Poisson(d.λ), x...)
logcdf(d::PoissonExpert, x...) = isinf(x...) ? 0.0 : Distributions.logcdf.(Distributions.Poisson(d.λ), x...)
cdf(d::PoissonExpert, x...) = isinf(x...) ? 1.0 : Distributions.cdf.(Distributions.Poisson(d.λ), x...)

## Parameters
params(d::PoissonExpert) = (d.λ)

## Simululation
sim_expert(d::PoissonExpert, sample_size) = Distributions.rand(Distributions.Poisson(d.λ), sample_size)

## penalty
penalty_init(d::PoissonExpert) = [2.0 1.0]
penalize(d::PoissonExpert, p) = (p[1]-1)*log(d.λ) - d.λ/p[2]