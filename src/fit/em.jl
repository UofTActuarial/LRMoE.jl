function EM_E_z_obs(gate_expert_ll_comp, gate_expert_ll)
    tmp = exp.(gate_expert_ll_comp .- gate_expert_ll)
    tmp[abs.(tmp) .< 1e-15] .= 0.0
    nan2num(tmp, 1 / size(gate_expert_ll_comp)[2])
    return tmp
end

function EM_E_z_lat(gate_expert_tn_bar_comp, gate_expert_tn_bar)
    tmp = exp.(gate_expert_tn_bar_comp .- gate_expert_tn_bar)
    tmp[abs.(tmp) .< 1e-15] .= 0.0
    nan2num(tmp, 1 / size(gate_expert_tn_bar_comp)[2])
    return tmp
end

# function EM_E_k(gate_expert_tn)
#     return expm1.( - gate_expert_tn )
# end

# function EM_E_k(gate_expert_tn_bar_k)
#     # return exp.( gate_expert_tn_bar_k )
#     return expm1.( - log1mexp.(gate_expert_tn_bar_k) )
# end

function EM_E_k_update(gate_expert_tn)
    return expm1(-gate_expert_tn) > 1e-15 ? expm1(-gate_expert_tn) : 0.0
end

function EM_E_k(gate_expert_tn)
    return EM_E_k_update.(-gate_expert_tn)
end

function EM_E_z_zero_obs_update(lower, prob, ll_vec)
    return lower == 0.0 ? prob / (prob + (1 - prob) * exp(ll_vec)) : 0.0
end

function EM_E_z_zero_lat_update(lower, prob, ll_vec)
    return lower > 0.0 ? prob / (prob + (1 - prob) * exp(ll_vec)) : 0.0
end

function EM_E_z_zero_obs(yl, p_old, gate_expert_ll_pos_comp)
    return EM_E_z_zero_obs_update.(yl, p_old, gate_expert_ll_pos_comp)
end

function EM_E_z_zero_lat(tl, p_old, gate_expert_tn_bar_pos_comp)
    return EM_E_z_zero_lat_update.(tl, p_old, gate_expert_tn_bar_pos_comp)
end

# EM

function EM_M_dQdα(X, comp_zkz_j, comp_zkz_marg, pp_j)
    return vec(sum(X .* (comp_zkz_j - comp_zkz_marg .* pp_j); dims=1))
end

function EM_M_dQ2dα2(X, comp_zkz_marg, pp_j, qq_j)
    return -X' * (comp_zkz_marg .* pp_j .* qq_j .* X)
end

function EM_M_α(X, α, z_e_obs, z_e_lat, k_e;
    α_iter_max=5, penalty=true, pen_α=5)
    let comp_zkz, comp_zkz_marg, α_new, α_old, iter
        # X, α_old, α_new, z_e_obs, z_e_lat, k_e, α_iter_max, penalty, pen_α, comp_zkz, comp_zkz_marg, iter

        comp_zkz = z_e_obs .+ (k_e .* z_e_lat)
        comp_zkz_marg = vec(sum(comp_zkz; dims=2))

        α_new = copy(α)
        α_old = copy(α_new) .- Inf

        iter = fill(0, size(α_new)[1])

        for j in 1:(size(α)[1] - 1)
            while (iter[j] < α_iter_max) & (sum((α_new[j, :] - α_old[j, :]) .^ 2) > 1e-08)
                α_old[j, :] = α_new[j, :]
                gate_body = X * α_new'
                pp = exp.(LogitGating(α_new, X))
                qqj = exp.(rowlogsumexp(gate_body[:, Not(j)]) - rowlogsumexp(gate_body))

                dQ =
                    EM_M_dQdα(X, comp_zkz[:, j], comp_zkz_marg, pp[:, j]) .-
                    (penalty ? vec(α_new[j, :] ./ pen_α^2) : 0.0)
                dQ2 =
                    EM_M_dQ2dα2(X, comp_zkz_marg, pp[:, j], qqj) - (
                        if penalty
                            (1.0 ./ pen_α^2) * I(size(α_new)[2])
                        else
                            (1e-07) * I(size(α_new)[2])
                        end
                    )

                α_new[j, :] = α_new[j, :] .- inv(dQ2) * dQ
                iter[j] = iter[j] + 1
            end
        end

        return α_new
    end
end

function EM_M_zero(z_zero_e_obs, z_pos_e_obs, z_zero_e_lat, z_pos_e_lat, k_e)
    num = sum(z_zero_e_obs .+ (z_zero_e_lat .* k_e))
    denom = num + sum(z_pos_e_obs .+ (z_pos_e_lat .* k_e))
    return num / denom
end