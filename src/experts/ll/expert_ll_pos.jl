
"""
    expert_ll_pos(d, tl, yl, yu, tu)

Function to calculate the loglikelihood of expert functions.

"""
# function expert_ll_pos(d::AnyExpert, tl::Real, yl::Real, yu::Real, tu::Real)
# 
# end

# expert_ll_pos(d, tl, yl. yu, tu) = _expert_ll_pos(d::AnyExpert, tl::Real, yl::Real, yu::Real, tu::Real)
# Broadcast.broadcasted(::typeof(expert_ll_pos), d, tl, yl, yu, tu) = broadcast(_expert_ll_pos, Ref(d), tl, yl, yu, tu)

## Non-zero inflated, continuous. e.g. LogNormal
function expert_ll_pos(d::ee, tl::Real, yl::Real, yu::Real, tu::Real) where {ee<:NonZIContinuousExpert}
    # function _expert_ll_pos(d::ee, tl::Real, yl::Real, yu::Real, tu::Real) where {ee<:NonZIContinuousExpert}

    # censor_idx = (yl .== yu)
    # expert_ll = fill(NaN, length(yu))
    # expert_ll[censor_idx] = logpdf.(d, yl[censor_idx])
    # expert_ll[.!censor_idx] = logcdf.(d, yu[.!censor_idx]) + log1mexp.(logcdf.(d, yl[.!censor_idx]) - logcdf.(d, yu[.!censor_idx]))

    # no_trunc_idx = (tl .== tu)
    # expert_tn = fill(NaN, length(tu))
    # expert_tn[no_trunc_idx] = logpdf.(d, tl[no_trunc_idx])
    # expert_tn[.!no_trunc_idx] = logcdf.(d, tu[.!no_trunc_idx]) + log1mexp.(logcdf.(d, tl[.!no_trunc_idx]) - logcdf.(d, tu[.!no_trunc_idx]))
    
    # zero_idx = (tu .== 0.0)
    # expert_ll[zero_idx] = -Inf
    # expert_tn[zero_idx] = -Inf
    
    # expert_tn_bar = fill(NaN, length(tu))
    # expert_tn[no_trunc_idx] = 0.0
    # expert_tn[.!no_trunc_idx] = log1mexp.(expert_tn[.!no_trunc_idx])

    # return (expert_ll = expert_ll, expert_tn = expert_tn, expert_tn_bar = expert_tn_bar)

    expert_ll = (yl == yu) ? logpdf.(d, yl) : logcdf.(d, yu) + log1mexp.(logcdf.(d, yl) - logcdf.(d, yu))
    expert_tn = (tl == tu) ? logpdf.(d, tl) : logcdf.(d, tu) + log1mexp.(logcdf.(d, tl) - logcdf.(d, tu))
    expert_ll = (tu == 0.) ? -Inf : expert_ll
    expert_tn = (tu == 0.) ? -Inf : expert_tn
    expert_tn_bar = log1mexp(expert_tn)

    # return (tl+yu)
    return [expert_ll, expert_tn, expert_tn_bar]
end

# expert_ll_pos(d, tl, yl, yu, tu) = _expert_ll_pos(d, tl, yl, yu, tu)
# Broadcast.broadcasted(::typeof(_expert_ll_pos), d, tl, yl, yu, tu) = broadcast(_expert_ll_pos, Ref{AnyExpert}(d), tl, yl, yu, tu)