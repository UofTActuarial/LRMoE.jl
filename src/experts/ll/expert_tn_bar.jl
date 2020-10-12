## Non-zero inflated, continuous. e.g. LogNormal
function expert_tn_bar(d::e, tl::Real, yl::Real, yu::Real, tu::Real) where {e<:NonZIContinuousExpert}
    # There is no zero inflation: only from the component
    return expert_tn_bar_pos(d, tl, yl, yu, tu)
end

## Zero inflated, continuous. e.g. ZILogNormal
function expert_tn_bar(d::e, tl::Real, yl::Real, yu::Real, tu::Real) where {e<:ZIContinuousExpert}
    # Possibly coming from the zero probability mass
    expert_tn_bar = (tl > 0.) ? log.(d.p + (1-d.p)*exp.(expert_tn_bar_pos(d, tl, yl, yu, tu))) : log.(0.0 + (1-d.p)*exp.(expert_tn_bar_pos(d, tl, yl, yu, tu)))

    return expert_tn_bar
end

## Non-zero inflated, discrete. e.g. Poisson
function expert_tn_bar(d::e, tl::Real, yl::Real, yu::Real, tu::Real) where {e<:NonZIDiscreteExpert}
    # There is no zero inflation: only from the component
    return expert_tn_bar_pos(d, tl, yl, yu, tu)
end

## Zero inflated, discrete. e.g. ZIPoisson
function expert_tn_bar(d::e, tl::Real, yl::Real, yu::Real, tu::Real) where {e<:ZIDiscreteExpert}
    # Possibly coming from the zero probability mass
    expert_tn_bar = (tl > 0.) ? log.(d.p + (1-d.p)*exp.(expert_tn_bar_pos(d, tl, yl, yu, tu))) : log.(0.0 + (1-d.p)*exp.(expert_tn_bar_pos(d, tl, yl, yu, tu)))

    return expert_tn_bar
end