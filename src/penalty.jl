function penalty_α(α, pen_α)
    return - 0.5 * sum(α.*α) / pen_α^2
end

function penalty_params(model, pen_params)
    return sum(sum([[penalize(model[j, k], pen_params[j][k]) for k in 1:size(model)[2]] for j in 1:size(model)[1]]))
end

# tmp = [[2. 1. 3. 1.], [2. 3. 4. 5.], [1. 2. 3. 4.]],
#        [[2. 1. 3. 1.], [2. 3. 4. 5.], [1. 2. 3. 5.]]

# [[penalize(model[j, k], tmp[j][k]) for k in 1:size(model)[2]] for j in 1:size(model)[1]]