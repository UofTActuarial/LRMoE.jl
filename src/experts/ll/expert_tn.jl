## Non-zero inflated, continuous. e.g. LogNormal
function expert_tn(d::e, tl::Real, yl::Real, yu::Real, tu::Real) where {e<:NonZIContinuousExpert}
    # There is no zero inflation: only from the component
    # return expert_tn_pos(d, tl, yl, yu, tu)

    # Possibly coming from the zero probability mass
    expert_tn = (tl == 0.) ? log.(0.0 + (1-0.0)*exp.(expert_tn_pos(d, tl, yl, yu, tu))) : log.(0.0 + (1-0.0)*exp.(expert_tn_pos(d, tl, yl, yu, tu)))
    # Must be from the zero probability mass
    expert_tn = (tu == 0.) ? log.(0.0) : expert_tn

    return expert_tn
end

## Zero inflated, continuous. e.g. ZILogNormal
function expert_tn(d::e, tl::Real, yl::Real, yu::Real, tu::Real) where {e<:ZIContinuousExpert}
    # Possibly coming from the zero probability mass
    p0 = params(d)[1]
    expert_tn = (tl == 0.) ? log.(p0 + (1-p0)*exp.(expert_tn_pos(d, tl, yl, yu, tu))) : log.(0.0 + (1-p0)*exp.(expert_tn_pos(d, tl, yl, yu, tu)))
    # Must be from the zero probability mass
    expert_tn = (tu == 0.) ? log.(p0) : expert_tn

    return expert_tn
end

## Non-zero inflated, discrete. e.g. Poisson
function expert_tn(d::e, tl::Real, yl::Real, yu::Real, tu::Real) where {e<:NonZIDiscreteExpert}
    # There is no zero inflation: only from the component
    return expert_tn_pos(d, tl, yl, yu, tu)
end

## Zero inflated, discrete. e.g. ZIPoisson
function expert_tn(d::e, tl::Real, yl::Real, yu::Real, tu::Real) where {e<:ZIDiscreteExpert}
    # Possibly coming from the zero probability mass
    p0 = params(d)[1]
    expert_tn = (tl == 0.) ? log.(p0 + (1-p0)*exp.(expert_tn_pos(d, tl, yl, yu, tu))) : log.(0.0 + (1-p0)*exp.(expert_tn_pos(d, tl, yl, yu, tu)))

    return expert_tn
end