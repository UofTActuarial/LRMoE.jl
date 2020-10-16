function expert_ll_pos_list(Y, model)
    # result = [[hcat([expert_ll_pos.(model[k, j], Y[:, 4*(k-1)+1], Y[:, 4*(k-1)+2], Y[:, 4*(k-1)+3], Y[:, 4*(k-1)+4]) for j in 1:size(model)[2]]...) for k in 1:size(model)[1]]...]
    return [[hcat([expert_ll_pos.(model[k, j], Y[:, 4*(j-1)+1], Y[:, 4*(j-1)+2], Y[:, 4*(j-1)+3], Y[:, 4*(j-1)+4]) for j in 1:size(model)[2]]...) for k in 1:size(model)[1]]...]
end

function expert_ll_list(Y, model)
    return [[hcat([expert_ll.(model[k, j], Y[:, 4*(j-1)+1], Y[:, 4*(j-1)+2], Y[:, 4*(j-1)+3], Y[:, 4*(j-1)+4]) for j in 1:size(model)[2]]...) for k in 1:size(model)[1]]...]
end

function expert_tn_pos_list(Y, model)
    return [[hcat([expert_tn_pos.(model[k, j], Y[:, 4*(j-1)+1], Y[:, 4*(j-1)+2], Y[:, 4*(j-1)+3], Y[:, 4*(j-1)+4]) for j in 1:size(model)[2]]...) for k in 1:size(model)[1]]...]
end

function expert_tn_list(Y, model)
    return [[hcat([expert_tn.(model[k, j], Y[:, 4*(j-1)+1], Y[:, 4*(j-1)+2], Y[:, 4*(j-1)+3], Y[:, 4*(j-1)+4]) for j in 1:size(model)[2]]...) for k in 1:size(model)[1]]...]
end

function expert_tn_bar_pos_list(Y, model)
    return [[hcat([expert_tn_bar_pos.(model[k, j], Y[:, 4*(j-1)+1], Y[:, 4*(j-1)+2], Y[:, 4*(j-1)+3], Y[:, 4*(j-1)+4]) for j in 1:size(model)[2]]...) for k in 1:size(model)[1]]...]
end

function expert_tn_bar_list(Y, model)
    return [[hcat([expert_tn_bar.(model[k, j], Y[:, 4*(j-1)+1], Y[:, 4*(j-1)+2], Y[:, 4*(j-1)+3], Y[:, 4*(j-1)+4]) for j in 1:size(model)[2]]...) for k in 1:size(model)[1]]...]
end