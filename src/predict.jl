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

"""
    predict_mean_prior(X, α, model)

Predicts the mean values of response, given covariates `X`, 
logit regression coefficients `α` and a specified `model` of expert functions.

# Arguments
- `X`: A matrix of covariates.
- `α`: A matrix of logit regression coefficients.
- `model`: A matrix specifying the expert functions.

# Return Values
- A matrix of predicted mean values of response, based on prior probabilities.
"""
function predict_mean_prior(X, α, model)
    weights = predict_class_prior(X, α).prob
    means = mean.(model)
    return [weights * means[j,:] for j in 1:size(means)[1]]
end

"""
    predict_mean_posterior(Y, X, α, model)

Predicts the mean values of response, observations `Y`, covariates `X`, 
logit regression coefficients `α` and a specified `model` of expert functions.

# Arguments
- `Y`: A matrix of responses.
- `X`: A matrix of covariates.
- `α`: A matrix of logit regression coefficients.
- `model`: A matrix specifying the expert functions.

# Return Values
- A matrix of predicted mean values of response, based on posterior probabilities.
"""
function predict_mean_posterior(Y, X, α, model)
    weights = predict_class_posterior(Y, X, α, model).prob
    means = mean.(model)
    return [weights * means[j,:] for j in 1:size(means)[1]]
end

"""
    predict_var_prior(X, α, model)

Predicts the variance of response, given covariates `X`, 
logit regression coefficients `α` and a specified `model` of expert functions.

# Arguments
- `X`: A matrix of covariates.
- `α`: A matrix of logit regression coefficients.
- `model`: A matrix specifying the expert functions.

# Return Values
- A matrix of predicted variance of response, based on prior probabilities.
"""
function predict_var_prior(X, α, model)
    weights = predict_class_prior(X, α).prob
    c_mean = mean.(model)
    g_mean = predict_mean_prior(X, α, model)
    var_c_mean = [vec(sum(hcat([(c_mean[d,j] .- g_mean[d]).^2 for j in 1:length(c_mean[d,:])]...) .* weights, dims = 2)) for d in 1:length(g_mean)]
    c_var = var.(model)
    mean_c_var = [weights * c_var[j,:] for j in 1:size(c_var)[1]]
    return [var_c_mean[j]+mean_c_var[j] for j in 1:length(var_c_mean)]
end

"""
    predict_var_posterior(Y, X, α, model)

Predicts the variance of response, observations `Y`, covariates `X`, 
logit regression coefficients `α` and a specified `model` of expert functions.

# Arguments
- `Y`: A matrix of responses.
- `X`: A matrix of covariates.
- `α`: A matrix of logit regression coefficients.
- `model`: A matrix specifying the expert functions.

# Return Values
- A matrix of predicted variance of response, based on posterior probabilities.
"""
function predict_var_posterior(Y, X, α, model)
    weights = predict_class_posterior(Y, X, α, model).prob
    c_mean = mean.(model)
    g_mean = predict_mean_prior(X, α, model)
    var_c_mean = [vec(sum(hcat([(c_mean[d,j] .- g_mean[d]).^2 for j in 1:length(c_mean[d,:])]...) .* weights, dims = 2)) for d in 1:length(g_mean)]
    c_var = var.(model)
    mean_c_var = [weights * c_var[j,:] for j in 1:size(c_var)[1]]
    return [var_c_mean[j]+mean_c_var[j] for j in 1:length(var_c_mean)]
end



# solve a quantile of a mixture model
# Bisection method seems to give the most stable results
function _solve_continuous_mix_quantile(weights, experts, p)
    p0 = sum(weights .* exp.(expert_ll.(experts, 0.0, 0.0, 0.0, Inf)))
    if p <= p0
        return 0.0
    else
        # init_guess = minimum([maximum(quantile.(experts, 0.90)) 500])
        # init_guess = maximum([maximum(quantile.(experts, 0.90)) 1000])
        init_guess = maximum([maximum(quantile.(experts, p)) 1000])
        VaR = try 
            # Roots.find_zero(y -> sum(weights .* exp.(expert_ll.(experts, 0.0, 0.0, y, Inf))) - p, init_guess, Roots.Order2())
            Roots.find_zero(y -> sum(weights .* exp.(expert_ll.(experts, 0.0, 0.0, y, Inf))) - p, (0.0, init_guess+500))
        catch;
            NaN
        end
        return VaR
    end
end

# calculate CTE based on VaR and p
# function _calc_continuous_mix_CTE(weights, experts, p, means, VaR)
#     return 
# end

"""
    predict_VaR_prior(X, α, model, p)

Predicts the `p`-th value-at-risk (VaR) of response, given covariates `X`, 
logit regression coefficients `α` and a specified `model` of expert functions.

# Arguments
- `X`: A matrix of covariates.
- `α`: A matrix of logit regression coefficients.
- `model`: A matrix specifying the expert functions.
- `p`: A vector of probabilities.

# Return Values
- A matrix of predicted variance of response, based on prior probabilities.
"""
function predict_VaR_prior(X, α, model, p)
    weights = predict_class_prior(X, α).prob
    # VaR = fill(NaN, size(X)[1], size(model)[1])
    # for i in 1:size(X)[1]
    #     for k in 1:size(model)[1]
    #         VaR[i,k] = try 
    #             find_zero(y -> sum(weights[i,:] .* exp.(expert_ll.(model[k,:], 0.0, 0.0, y, Inf))) - p, 100)
    #         catch;
    #             NaN
    #         end
    #     end
    # end
    # return VaR
    return vcat([hcat([_solve_continuous_mix_quantile(weights[i,:], model[k,:], p) for k in 1:size(model)[1] ]...) for i in 1:size(X)[1]]...)
end



