"""
    predict_class_prior(X, α)

Predicts the latent class probabilities, given covariates `X` and logit regression coefficients `α`.

# Arguments
- `X`: A matrix of covariates.
- `α`: A matrix of logit regression coefficients.

# Return Values
- `prob`: A matrix of latent class probabilities.
- `max_prob_idx`: A matrix of the most likely latent class for each observation.
"""
function predict_class_prior(X, α)
    tmp = exp.(LogitGating(α, X))
    return (prob = tmp, max_prob_idx = [findmax(tmp[i,:])[2] for i in 1:size(tmp)[1]])
end


"""
    predict_class_posterior(Y, X, α, model)

Predicts the latent class probabilities, given observations `Y`, covariates `X`, 
logit regression coefficients `α` and a specified `model` of expert functions. 

# Arguments
- `Y`: A matrix of responses.
- `X`: A matrix of covariates.
- `α`: A matrix of logit regression coefficients.
- `model`: A matrix specifying the expert functions.

# Return Values
- `prob`: A matrix of latent class probabilities.
- `max_prob_idx`: A matrix of the most likely latent class for each observation.
"""
function predict_class_posterior(Y, X, α, model)
    gate = LogitGating(α, X)
    ll_np_list = loglik_np(Y, gate, model)
    z_e_obs = EM_E_z_obs(ll_np_list.gate_expert_ll_comp, ll_np_list.gate_expert_ll)
    return (prob = z_e_obs, max_prob_idx = [findmax(z_e_obs[i,:])[2] for i in 1:size(z_e_obs)[1]])
end

function predict_mean_prior(X, α, model)
    weights = predict_class_prior(X, α).prob
    means = mean.(model)
    return [weights * means[j,:] for j in 1:size(means)[1]]
end

function predict_mean_posterior(Y, X, α, model)
    weights = predict_class_posterior(Y, X, α, model).prob
    means = mean.(model)
    return [weights * means[j,:] for j in 1:size(means)[1]]
end

function predict_var_prior(X, α, model)
    weights = predict_class_prior(X, α).prob
    c_mean = mean.(model)
    g_mean = predict_mean_prior(X, α, model)
    var_c_mean = [vec(sum(hcat([(c_mean[d,j] .- g_mean[d]).^2 for j in 1:length(c_mean[d,:])]...) .* weights, dims = 2)) for d in 1:length(g_mean)]
    c_var = var.(model)
    mean_c_var = [weights * c_var[j,:] for j in 1:size(c_var)[1]]
    return [var_c_mean[j]+mean_c_var[j] for j in 1:length(var_c_mean)]
end

function predict_var_posterior(Y, X, α, model)
    weights = predict_class_posterior(Y, X, α, model).prob
    c_mean = mean.(model)
    g_mean = predict_mean_prior(X, α, model)
    var_c_mean = [vec(sum(hcat([(c_mean[d,j] .- g_mean[d]).^2 for j in 1:length(c_mean[d,:])]...) .* weights, dims = 2)) for d in 1:length(g_mean)]
    c_var = var.(model)
    mean_c_var = [weights * c_var[j,:] for j in 1:size(c_var)[1]]
    return [var_c_mean[j]+mean_c_var[j] for j in 1:length(var_c_mean)]
end


