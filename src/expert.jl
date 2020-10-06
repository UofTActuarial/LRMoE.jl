const DiscreteExpert = DiscreteUnivariateDistribution
const ContinuousExpert = ContinuousUnivariateDistribution

function expert_ll(d::DiscreteExpert, yl::Real, yu::Real)
    censor_idx = yl .== yu
    result = fill(NaN, length(yu))
    result[censor_idx] = logpdf.(d, yl[censor_idx])
    result[.!censor_idx] = logcdf.(d, yu[.!censor_idx]) + log1mexp.(logcdf.(d, yl[.!censor_idx]) - logcdf.(d, yu[.!censor_idx]))

    return result
end