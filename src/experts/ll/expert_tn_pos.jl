# ## Non-zero inflated, continuous. e.g. LogNormal
# function expert_tn_pos(d::e, tl::Real, yl::Real, yu::Real, tu::Real) where {e<:NonZIContinuousExpert}

#     expert_tn = (tl == tu) ? logpdf.(d, tl) : logcdf.(d, tu) + log1mexp.(logcdf.(d, tl) - logcdf.(d, tu))
#     expert_tn = (tu == 0.) ? -Inf : expert_tn

#     return expert_tn
# end

# ## Zero inflated, continuous. e.g. ZILogNormal
# function expert_tn_pos(d::e, tl::Real, yl::Real, yu::Real, tu::Real) where {e<:ZIContinuousExpert}

#     expert_tn = (tl == tu) ? logpdf.(d, tl) : logcdf.(d, tu) + log1mexp.(logcdf.(d, tl) - logcdf.(d, tu))
#     expert_tn = (tu == 0.) ? -Inf : expert_tn

#     return expert_tn
# end

# ## Non-zero inflated, discrete. e.g. Poisson
# function expert_tn_pos(d::e, tl::Real, yl::Real, yu::Real, tu::Real) where {e<:NonZIDiscreteExpert}

#     expert_tn = (tl == tu) ? logpdf.(d, tl) : logcdf.(d, tu) + log1mexp.(logcdf.(d, ceil.(tl) .- 1) - logcdf.(d, tu))

#     return expert_tn
# end

# ## Zero inflated, discrete. e.g. ZIPoisson
# function expert_tn_pos(d::e, tl::Real, yl::Real, yu::Real, tu::Real) where {e<:ZIDiscreteExpert}

#     expert_tn = (tl == tu) ? logpdf.(d, tl) : logcdf.(d, tu) + log1mexp.(logcdf.(d, ceil.(tl) .- 1) - logcdf.(d, tu))

#     return expert_tn
# end