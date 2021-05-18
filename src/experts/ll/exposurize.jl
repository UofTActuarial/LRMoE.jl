function exposurize_model(model; exposure = 1)
    return cat([LRMoE.exposurize_expert.(model, exposure = exposure[i]) for i in 1:length(exposure)]..., dims = 3)
end

# Exact case
function expert_ll_ind_mat_exact(Y, model)
    return cat([[LRMoE.expert_ll_exact.(model[j, k], Y[j]) for j in 1:size(model)[1]] for k in 1:size(model)[2]]..., dims = 2)
end

function expert_ll_list_exact(Y, model)
    return cat([expert_ll_ind_mat_exact(Y[i,:], model[:,:,i]) for i in 1:size(Y)[1]]..., dims = 3)
end