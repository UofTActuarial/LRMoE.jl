# j: counts dimension of Y
# k: counts component
function expert_ll_ind_mat(Y, model)
    result = fill(NaN, size(model)[1], size(model)[2])
    for j in 1:size(model)[1]
        for k in 1:size(model)[2]
            result[j, k] =
                LRMoE.expert_ll.(
                    model[j, k],
                    Y[4 * (j - 1) + 1],
                    Y[4 * (j - 1) + 2],
                    Y[4 * (j - 1) + 3],
                    Y[4 * (j - 1) + 4],
                )
        end
    end
    return result
end

function expert_ll_list(Y, model)
    result = fill(NaN, size(model)[1], size(model)[2], size(Y)[1])
    for i in 1:size(Y)[1]
        result[:, :, i] = expert_ll_ind_mat(Y[i, :], model[:, :, i])
    end
    return result
end

function expert_tn_ind_mat(Y, model)
    result = fill(NaN, size(model)[1], size(model)[2])
    for j in 1:size(model)[1]
        for k in 1:size(model)[2]
            result[j, k] =
                LRMoE.expert_tn.(
                    model[j, k],
                    Y[4 * (j - 1) + 1],
                    Y[4 * (j - 1) + 2],
                    Y[4 * (j - 1) + 3],
                    Y[4 * (j - 1) + 4],
                )
        end
    end
    return result
end

function expert_tn_list(Y, model)
    result = fill(NaN, size(model)[1], size(model)[2], size(Y)[1])
    for i in 1:size(Y)[1]
        result[:, :, i] = expert_tn_ind_mat(Y[i, :], model[:, :, i])
    end
    return result
end

function expert_tn_bar_ind_mat(Y, model)
    result = fill(NaN, size(model)[1], size(model)[2])
    for j in 1:size(model)[1]
        for k in 1:size(model)[2]
            result[j, k] =
                LRMoE.expert_tn_bar.(
                    model[j, k],
                    Y[4 * (j - 1) + 1],
                    Y[4 * (j - 1) + 2],
                    Y[4 * (j - 1) + 3],
                    Y[4 * (j - 1) + 4],
                )
        end
    end
    return result
end

function expert_tn_bar_list(Y, model)
    result = fill(NaN, size(model)[1], size(model)[2], size(Y)[1])
    for i in 1:size(Y)[1]
        result[:, :, i] = expert_tn_bar_ind_mat(Y[i, :], model[:, :, i])
    end
    return result
end

function loglik_np(Y, gate, model)

    # By dimension, then by component
    expert_ll_dim_comp = expert_ll_list(Y, model)
    expert_tn_dim_comp = expert_tn_list(Y, model)
    expert_tn_bar_dim_comp = expert_tn_bar_list(Y, model)

    # Aggregate by dimension
    expert_ll_comp = loglik_aggre_dim(expert_ll_dim_comp)
    expert_tn_comp = loglik_aggre_dim(expert_tn_dim_comp)
    expert_tn_bar_comp = loglik_aggre_dim(expert_tn_bar_dim_comp)

    # Adding the gating function
    gate_expert_ll_comp = loglik_aggre_gate_dim(gate, expert_ll_comp)
    gate_expert_tn_comp = loglik_aggre_gate_dim(gate, expert_tn_comp)
    gate_expert_tn_bar_comp = gate + log1mexp.(expert_tn_comp)
    gate_expert_tn_bar_comp_k = gate + log1mexp.(expert_tn_bar_comp)
    gate_expert_tn_bar_comp_z_lat = loglik_aggre_gate_dim(gate, expert_tn_bar_comp)

    # Aggregate by component
    gate_expert_ll = loglik_aggre_gate_dim_comp(gate_expert_ll_comp)
    gate_expert_tn = loglik_aggre_gate_dim_comp(gate_expert_tn_comp)
    gate_expert_tn_bar = loglik_aggre_gate_dim_comp(gate_expert_tn_bar_comp)
    gate_expert_tn_bar_k = loglik_aggre_gate_dim_comp(gate_expert_tn_bar_comp_k)
    gate_expert_tn_bar_z_lat = loglik_aggre_gate_dim_comp(gate_expert_tn_bar_comp_z_lat)

    # Normalize by tn & tn_bar
    norm_gate_expert_ll = gate_expert_ll - gate_expert_tn_bar_k

    # Sum over all observations
    ll = sum(norm_gate_expert_ll)

    return (
        expert_tn_comp=expert_tn_comp,
        expert_tn_bar_comp=expert_tn_bar_comp, gate_expert_ll_comp=gate_expert_ll_comp,
        gate_expert_tn_comp=gate_expert_tn_comp,
        gate_expert_tn_bar_comp=gate_expert_tn_bar_comp,
        gate_expert_tn_bar_comp_k=gate_expert_tn_bar_comp_k,
        gate_expert_tn_bar_comp_z_lat=gate_expert_tn_bar_comp_z_lat,
        gate_expert_ll=gate_expert_ll,
        gate_expert_tn=gate_expert_tn,
        gate_expert_tn_bar=gate_expert_tn_bar,
        gate_expert_tn_bar_k=gate_expert_tn_bar_k,
        gate_expert_tn_bar_z_lat=gate_expert_tn_bar_z_lat,
        norm_gate_expert_ll=norm_gate_expert_ll, ll=ll,
    )
end