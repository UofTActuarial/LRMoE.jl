# struct LogitGating{T<:Array{Union{Nothing, Number}}} <: NonRandomGating
#     α::T
#     LogitGating{T}(α::T) where {T<:Array{Union{Nothing, Number}}} = new{T}(α::T)
# end

# function LogitGating(α::T; check_args=true) where {T <: Array{Union{Nothing, Number}}}
#     check_args # && @check_args(LogitGating, isa.(α, Number))
#     return LogitGating{T}(α)
# end

#### Outer constructors
# LogitGating(α::Array{Union{Nothing, Real}}) = LogitGating(promote(α)...)
# LogitGating(α::Array{Union{Nothing, Integer}}) = LogitGating(float(α))

# SHOULD BE ARRAY TYPE

function LogitGating(α, x; check_args=true)
    check_args && @check_args(LogitGating, size(α)[2] == size(x)[2])
    ax = x * α'
    rowsum = rowlogsumexp(ax)
    return ax .- rowsum
end