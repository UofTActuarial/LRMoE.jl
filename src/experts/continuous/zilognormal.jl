"""
    ZILogNormalExpert(p, μ, σ)

Expert function: `ZILogNormalExpert(p, μ, σ)`.

"""
struct ZILogNormalExpert{T<:Real} <: ZIContinuousExpert
    p::T
    μ::T
    σ::T
end


function ZILogNormalExpert(p::T, μ::T, σ::T; check_args=true) where {T <: Real}
    check_args && @check_args(ZILogNormalExpert, 0 <= p <= 1 && σ >= zero(σ))
    return ZILogNormalExpert{T}(p, μ, σ)
end


#### Outer constructors
ZILogNormalExpert(p::Real, μ::Real, σ::Real) = ZILogNormalExpert(promote(p, μ, σ)...)
ZILogNormalExpert(p::Integer, μ::Integer, σ::Integer) = ZILogNormalExpert(float(p), float(μ), float(σ))