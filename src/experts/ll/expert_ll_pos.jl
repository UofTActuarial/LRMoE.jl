
"""
    expert_ll_pos(d, tl, yl, yu, tu)

Function to calculate the loglikelihood of expert functions.

"""
function expert_ll_pos(d::AnyExpert, tl::Real, yl::Real, yu::Real, tu::Real)

end

## Non-zero inflated, continuous. e.g. LogNormal
function expert_ll_pos(d::NonZIContinuousExpert, tl::Real, yl::Real, yu::Real, tu::Real)
    censor_idx = (yl .== yu)
    result = fill(NaN, length(yu))
    result[censor_idx] = logpdf.(d, yl[censor_idx])
    result[.!censor_idx] = logcdf.(d, yu[.!censor_idx]) + log1mexp.(logcdf.(d, yl[.!censor_idx]) - logcdf.(d, yu[.!censor_idx]))

    return result
end