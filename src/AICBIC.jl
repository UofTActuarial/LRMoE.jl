function _count_α(α)
    return (size(α)[1] - 1) * size(α)[2]
end

function _count_params(model)
    return sum(length.(params.(model)))
end
