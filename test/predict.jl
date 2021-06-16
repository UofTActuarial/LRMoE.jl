# using Test
# using Distributions
# using StatsFuns

# using LRMoE


# # Poisson
# param0 = rand(Distributions.Uniform(0, 1), 1)[1]
# param1 = rand(Distributions.Uniform(0.5, 8), 1)[1]

# X = rand(Distributions.Uniform(-5, 5), 20000, 7)
# α = rand(Distributions.Uniform(-1, 1), 3, 7)
# α[2,:] .= 0

# model = [PoissonExpert(param1) PoissonExpert(2*param1) PoissonExpert(3*param1);
#         PoissonExpert(3*param1) PoissonExpert(0.5*param1) PoissonExpert(1.5*param1)]

# expos = rand(Distributions.Uniform(0.1, 5), 20000)
# expos_future = ceil.(expos) .- expos
# Y_sim = LRMoE.sim_dataset(α, X, model, exposure = expos)

# classes = LRMoE.predict_class_prior(X, α)
# classes.prob
# classes.max_prob_idx

# classes = LRMoE.predict_class_posterior(Y_sim, X, α, model, exposure_past = expos)
# classes.prob
# classes.max_prob_idx

# m1 = LRMoE.predict_mean_prior(X, α, model, exposure_future = expos_future)
# m2 = LRMoE.predict_mean_posterior(Y_sim, X, α, model, exposure_past = expos, exposure_future = expos_future)

# minimum(m2-m1, dims = 1)
# maximum(m2-m1, dims = 1)

# v1 = LRMoE.predict_var_prior(X, α, model, exposure_future = expos_future)
# v2 = LRMoE.predict_var_posterior(Y_sim, X, α, model, exposure_past = expos, exposure_future = expos_future)


# minimum(v2-v1, dims = 1)
# maximum(v2-v1, dims = 1)


# weights = classes.prob
# model_exp = LRMoE.exposurize_model(model, exposure = expos)
# means = mean.(model_exp)

# result = fill(NaN, size(weights)[1], 2)

# c_mean = mean.(model_exp)
# g_mean = predict_mean_prior(X, α, model, exposure = expos)
# var_c_mean = fill(NaN, size(X)[1], size(model)[2])
# for i in 1:size(X)[1]
#     var_c_mean[i,:] = (c_mean[:,:,i] .- g_mean[i,:] ).^2 * weights[i,:]
# end
# c_var = var.(model_exp)
# mean_c_var = fill(NaN, size(X)[1], size(model)[2])
# for i in 1:size(X)[1]
#     mean_c_var[i,:] = c_var[:,:,i] * weights[i,:]
# end

# c_mean[:,:,1]
# g_mean[1,:]
# (c_mean[:,:,1] .- g_mean[1,:]) * weights[1,:]

# c_var[:,:,1] * weights[1,:]




# model = [LogNormalExpert(param1, 1) LogNormalExpert(2*param1, 3) LogNormalExpert(3*param1, 2);
#         LogNormalExpert(3*param1, 0.5) LogNormalExpert(0.5*param1, 1.5) LogNormalExpert(1.5*param1, 1)]

# expos = rand(Distributions.Uniform(0.1, 5), 20000)
# expos_future = ceil.(expos) .- expos
# Y_sim = LRMoE.sim_dataset(α, X, model, exposure = expos)

# lim = rand(Uniform(0.1, 10), size(X)[1], 2)
# tmp1 = LRMoE.predict_limit_prior(X, α, model, lim, exposure_future = expos_future)
# tmp2 = LRMoE.predict_excess_prior(X, α, model, lim, exposure_future = expos_future)
# m = LRMoE.predict_mean_prior(X, α, model, exposure_future = expos_future)
# sum(m - tmp1 - tmp2)

# tmp1 = LRMoE.predict_limit_posterior(Y_sim, X, α, model, lim, exposure_past = expos, exposure_future = expos_future)
# tmp2 = LRMoE.predict_excess_posterior(Y_sim, X, α, model, lim, exposure_past = expos, exposure_future = expos_future)
# m = LRMoE.predict_mean_posterior(Y_sim, X, α, model, exposure_past = expos, exposure_future = expos_future)
# sum(m - tmp1 - tmp2)

# qs = rand(Uniform(0, 1), size(X)[1], 2)
# tmp1 = LRMoE.predict_VaRCTE_prior(X, α, model, qs, exposure_future = expos_future)
# tmp2 = LRMoE.predict_VaRCTE_posterior(Y_sim, X, α, model, qs, exposure_past = expos, exposure_future = expos_future)




# for i in 1:size(weights)[1]
#     result[i,:] = means[:,:,i] * weights[i,:]
# end

# hcat([means[:,:,i] * weights[i,:] for i in 1:size(weights)[1]]...)

# size(means)

# means[:,:,1]

# weights[1,:]



# gate = LogitGating(α, X)

# function _exact_to_full(Y)
#     result = fill(NaN, size(Y)[1], size(Y)[2]*4)
#     for j in 1:size(Y)[2]
#         result[:,4*(j-1)+1] .= 0.0
#         result[:,4*(j-1)+2] .= Y[:,j]
#         result[:,4*(j-1)+3] .= Y[:,j]
#         result[:,4*(j-1)+4] .= Inf
#     end
#     return result
# end

# Y_full = _exact_to_full(Y_sim)
# ll_np_list = LRMoE.loglik_np(Y_full, gate, model_exp)

# z_e_obs = LRMoE.EM_E_z_obs(ll_np_list.gate_expert_ll_comp, ll_np_list.gate_expert_ll)

# tmp = mean.(model_exp)