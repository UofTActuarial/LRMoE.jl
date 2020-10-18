function nan2num(x, g)
    for i in eachindex(x)
        @inbounds x[i] = ifelse(isnan(x[i]), g, x[i])
    end
end

function EM_E_z_obs(gate_expert_ll_comp, gate_expert_ll)
    return exp.(gate_expert_ll_comp .- gate_expert_ll)
end

function EM_E_z_lat(gate_expert_tn_bar_comp, gate_expert_tn_bar)
    tmp = exp.(gate_expert_tn_bar_comp .- gate_expert_tn_bar)
    nan2num(tmp, 1/size(gate_expert_tn_bar_comp)[2])
        # Slower: # tmp[isnan.(tmp)] .= 1/size(gate_expert_tn_bar_comp)[2]
    return tmp
end

function EM_E_k(gate_expert_tn)
    return expm1.( - gate_expert_tn )
end