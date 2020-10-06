"""
    LogNormalExpert(μ, σ)

Expert function: `LogNormal(μ, σ)` and `ZILogNormalExpert(p, μ, σ)`.

"""
struct LogNormalExpert{T<:Real} <: NonZIContinuousExpert
    μ::T
    σ::T
end

function LogNormalExpert(μ::T, σ::T; check_args=true) where {T <: Real}
    check_args && @check_args(LogNormalExpert, σ >= zero(σ))
    return LogNormalExpert{T}(μ, σ)
end

#### Outer constructors
LogNormalExpert(μ::Real, σ::Real) = LogNormalExpert(promote(μ, σ)...)
LogNormalExpert(μ::Integer, σ::Integer) = LogNormalExpert(float(μ), float(σ))