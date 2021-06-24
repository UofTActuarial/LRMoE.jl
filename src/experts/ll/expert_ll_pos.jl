# ## Non-zero inflated, continuous. e.g. LogNormal
# function expert_ll_pos(d::e, tl::Real, yl::Real, yu::Real, tu::Real) where {e<:NonZIContinuousExpert}

#     expert_ll = (yl == yu) ? logpdf.(d, yl) : logcdf.(d, yu) + log1mexp.(logcdf.(d, yl) - logcdf.(d, yu))
#     expert_ll = (tu == 0.) ? -Inf : expert_ll

#     return expert_ll
# end

# ## Zero inflated, continuous. e.g. ZILogNormal
# function expert_ll_pos(d::e, tl::Real, yl::Real, yu::Real, tu::Real) where {e<:ZIContinuousExpert}

#     expert_ll = (yl == yu) ? logpdf.(d, yl) : logcdf.(d, yu) + log1mexp.(logcdf.(d, yl) - logcdf.(d, yu))
#     expert_ll = (tu == 0.) ? -Inf : expert_ll

#     return expert_ll
# end

# ## Non-zero inflated, discrete. e.g. Poisson
# function expert_ll_pos(d::e, tl::Real, yl::Real, yu::Real, tu::Real) where {e<:NonZIDiscreteExpert}

#     expert_ll = (yl == yu) ? logpdf.(d, yl) : logcdf.(d, yu) + log1mexp.(logcdf.(d, ceil.(yl) .- 1) - logcdf.(d, yu))

#     return expert_ll
# end

# ## Zero inflated, discrete. e.g. ZIPoisson
# function expert_ll_pos(d::e, tl::Real, yl::Real, yu::Real, tu::Real) where {e<:ZIDiscreteExpert}

#     expert_ll = (yl == yu) ? logpdf.(d, yl) : logcdf.(d, yu) + log1mexp.(logcdf.(d, ceil.(yl) .- 1) - logcdf.(d, yu))

#     return expert_ll
# end