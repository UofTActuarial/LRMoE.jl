## Non-zero inflated, continuous. e.g. LogNormal
function expert_tn_bar_pos(d::e, expert_tn::Real) where {e<:NonZIContinuousExpert}

    expert_tn_bar = log1mexp.(expert_tn)

    return expert_tn_bar
end

## Zero inflated, continuous. e.g. ZILogNormal
function expert_tn_bar_pos(d::e, expert_tn::Real) where {e<:ZIContinuousExpert}

    expert_tn_bar = log1mexp.(expert_tn)

    return expert_tn_bar
end

