## Non-zero inflated, continuous. e.g. LogNormal
function expert_tn_bar_pos(d::e, tl::Real, yl::Real, yu::Real, tu::Real) where {e<:NonZIContinuousExpert}

    expert_tn_bar = (tl == tu) ? 0.0 : log1mexp.(logcdf.(d, tu) + log1mexp.(logcdf.(d, tl) - logcdf.(d, tu)))

    return expert_tn_bar
end

## Zero inflated, continuous. e.g. ZILogNormal
function expert_tn_bar_pos(d::e, tl::Real, yl::Real, yu::Real, tu::Real) where {e<:ZIContinuousExpert}

    expert_tn_bar = (tl == tu) ? 0.0 : log1mexp.(logcdf.(d, tu) + log1mexp.(logcdf.(d, tl) - logcdf.(d, tu)))

    return expert_tn_bar
end

