function exposurize_model(model; exposure=1)
    result = Array{Union{Nothing,LRMoE.AnyExpert}}(
        nothing, size(model)[1], size(model)[2], length(exposure)
    )
    for i in 1:length(exposure)
        result[:, :, i] = LRMoE.exposurize_expert.(model, exposure=exposure[i])
    end
    return result
end

function loglik_aggre_dim(ls)
    return Matrix(sum(ls; dims=1)[1, :, :]')
end

function loglik_aggre_gate_dim(gate, ll_mat)
    return gate + ll_mat
end

function loglik_aggre_gate_dim_comp(ll_mat)
    return rowlogsumexp(ll_mat)
end

# Exact case
function expert_ll_ind_mat_exact(Y, model)
    result = fill(NaN, size(model)[1], size(model)[2])
    for j in 1:size(model)[1]
        for k in 1:size(model)[2]
            result[j, k] = LRMoE.expert_ll_exact.(model[j, k], Y[j])
        end
    end
    return result # cat([[LRMoE.expert_ll_exact.(model[j, k], Y[j]) for j in 1:size(model)[1]] for k in 1:size(model)[2]]..., dims = 2)
end

function expert_ll_list_exact(Y, model)
    result = fill(NaN, size(model)[1], size(model)[2], size(Y)[1])
    for i in 1:size(Y)[1]
        result[:, :, i] = expert_ll_ind_mat_exact(Y[i, :], model[:, :, i])
    end
    return result # cat([expert_ll_ind_mat_exact(Y[i,:], model[:,:,i]) for i in 1:size(Y)[1]]..., dims = 3)
end
