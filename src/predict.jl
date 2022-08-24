"""
    predict_class_prior(X, α)

Predicts the latent class probabilities, 
given covariates `X` 
and logit regression coefficients `α`.

# Arguments
- `X`: A matrix of covariates.
- `α`: A matrix of logit regression coefficients.

# Return Values
- `prob`: A matrix of latent class probabilities.
- `max_prob_idx`: A matrix of the most likely latent class for each observation.
"""
function predict_class_prior(X, α)
    tmp = exp.(LogitGating(α, X))
    return (prob=tmp, max_prob_idx=[findmax(tmp[i, :])[2] for i in 1:size(tmp)[1]])
end

"""
    predict_class_posterior(Y, X, α, model; 
        exact_Y = true, exposure_past = nothing)

Predicts the latent class probabilities, 
given observations `Y`, covariates `X`, 
logit regression coefficients `α` and a specified `model` of expert functions. 

# Arguments
- `Y`: A matrix of responses.
- `X`: A matrix of covariates.
- `α`: A matrix of logit regression coefficients.
- `model`: A matrix specifying the expert functions.

# Optional Arguments
- `exact_Y`: `true` or `false` (default), indicating if `Y` is observed exactly or with censoring and truncation.
- `exposure_past`: A vector indicating the time exposure (past) of each observation. If nothing is supplied, it is set to 1.0 by default.

# Return Values
- `prob`: A matrix of latent class probabilities.
- `max_prob_idx`: A matrix of the most likely latent class for each observation.
"""
function predict_class_posterior(Y, X, α, model; exact_Y=true, exposure_past=nothing)
    if exact_Y == true
        Y = _exact_to_full(Y)
    end

    if isnothing(exposure_past)
        exposure_past = fill(1.0, size(X)[1])
    end

    model_exp = exposurize_model(model; exposure=exposure_past)

    gate = LogitGating(α, X)
    ll_np_list = loglik_np(Y, gate, model_exp)
    z_e_obs = EM_E_z_obs(ll_np_list.gate_expert_ll_comp, ll_np_list.gate_expert_ll)
    return (
        prob=z_e_obs, max_prob_idx=[findmax(z_e_obs[i, :])[2] for i in 1:size(z_e_obs)[1]]
    )
end

"""
    predict_mean_prior(X, α, model; 
        exposure_future = nothing)

Predicts the mean values of response, 
given covariates `X`, 
logit regression coefficients `α` and a specified `model` of expert functions.

# Arguments
- `X`: A matrix of covariates.
- `α`: A matrix of logit regression coefficients.
- `model`: A matrix specifying the expert functions.

# Optional Arguments
- `exposure_future`: A vector indicating the time exposure (future) of each observation. If nothing is supplied, it is set to 1.0 by default.

# Return Values
- A matrix of predicted mean values of response, based on prior probabilities.
"""
function predict_mean_prior(X, α, model; exposure_future=nothing)
    if isnothing(exposure_future)
        exposure_future = fill(1.0, size(X)[1])
    end

    model_exp = exposurize_model(model; exposure=exposure_future)

    weights = predict_class_prior(X, α).prob
    means = mean.(model_exp)

    result = fill(NaN, size(X)[1], size(model)[1])

    for i in 1:size(X)[1]
        result[i, :] = means[:, :, i] * weights[i, :]
    end

    return result
end

"""
    predict_mean_posterior(Y, X, α, model; 
        exact_Y = true, exposure_past = nothing, exposure_future = nothing)

Predicts the mean values of response,
given observations `Y`, covariates `X`, 
logit regression coefficients `α` and a specified `model` of expert functions.

# Arguments
- `Y`: A matrix of responses.
- `X`: A matrix of covariates.
- `α`: A matrix of logit regression coefficients.
- `model`: A matrix specifying the expert functions.

# Optional Arguments
- `exact_Y`: `true` or `false` (default), indicating if `Y` is observed exactly or with censoring and truncation.
- `exposure_past`: A vector indicating the time exposure (past) of each observation. If nothing is supplied, it is set to 1.0 by default.
- `exposure_future`: A vector indicating the time exposure (future) of each observation. If nothing is supplied, it is set to 1.0 by default.

# Return Values
- A matrix of predicted mean values of response, based on posterior probabilities.
"""
function predict_mean_posterior(
    Y, X, α, model; exact_Y=true, exposure_past=nothing, exposure_future=nothing
)
    # if exact_Y == true
    #     Y = _exact_to_full(Y)
    # end

    if isnothing(exposure_past)
        exposure_past = fill(1.0, size(X)[1])
    end

    if isnothing(exposure_future)
        exposure_future = fill(1.0, size(X)[1])
    end

    model_exp = exposurize_model(model; exposure=exposure_future)

    weights =
        predict_class_posterior(
            Y, X, α, model; exact_Y=exact_Y, exposure_past=exposure_past
        ).prob
    means = mean.(model_exp)

    result = fill(NaN, size(X)[1], size(model)[1])

    for i in 1:size(X)[1]
        result[i, :] = means[:, :, i] * weights[i, :]
    end

    return result
end

"""
    predict_var_prior(X, α, model; 
        exposure_future = nothing)

Predicts the variance of response, 
given covariates `X`, 
logit regression coefficients `α` and a specified `model` of expert functions.

# Arguments
- `X`: A matrix of covariates.
- `α`: A matrix of logit regression coefficients.
- `model`: A matrix specifying the expert functions.

# Optional Arguments
- `exposure_future`: A vector indicating the time exposure of each observation. If nothing is supplied, it is set to 1.0 by default.

# Return Values
- A matrix of predicted variance of response, based on prior probabilities.
"""
function predict_var_prior(X, α, model; exposure_future=nothing)
    if isnothing(exposure_future)
        exposure_future = fill(1.0, size(X)[1])
    end

    model_exp = exposurize_model(model; exposure=exposure_future)

    weights = predict_class_prior(X, α).prob

    c_mean = mean.(model_exp)
    g_mean = predict_mean_prior(X, α, model; exposure_future=exposure_future)
    c_var = var.(model_exp)

    var_c_mean = fill(NaN, size(X)[1], size(model)[1])
    mean_c_var = fill(NaN, size(X)[1], size(model)[1])

    for i in 1:size(X)[1]
        var_c_mean[i, :] = (c_mean[:, :, i] .- g_mean[i, :]) .^ 2 * weights[i, :]
        mean_c_var[i, :] = c_var[:, :, i] * weights[i, :]
    end

    return var_c_mean + mean_c_var
end

"""
    predict_var_posterior(Y, X, α, model; 
        exact_Y = true, exposure_past = nothing, exposure_future = nothing)

Predicts the variance of response, 
given observations `Y`, covariates `X`, 
logit regression coefficients `α` and a specified `model` of expert functions.

# Arguments
- `Y`: A matrix of responses.
- `X`: A matrix of covariates.
- `α`: A matrix of logit regression coefficients.
- `model`: A matrix specifying the expert functions.

# Optional Arguments
- `exact_Y`: `true` or `false` (default), indicating if `Y` is observed exactly or with censoring and truncation.
- `exposure_past`: A vector indicating the time exposure (past) of each observation. If nothing is supplied, it is set to 1.0 by default.
- `exposure_future`: A vector indicating the time exposure (future) of each observation. If nothing is supplied, it is set to 1.0 by default.

# Return Values
- A matrix of predicted variance of response, based on posterior probabilities.
"""
function predict_var_posterior(
    Y, X, α, model; exact_Y=true, exposure_past=nothing, exposure_future=nothing
)
    # if exact_Y == true
    #     Y = _exact_to_full(Y)
    # end

    if isnothing(exposure_past)
        exposure_past = fill(1.0, size(X)[1])
    end

    if isnothing(exposure_future)
        exposure_future = fill(1.0, size(X)[1])
    end

    model_exp = exposurize_model(model; exposure=exposure_future)

    weights =
        predict_class_posterior(
            Y, X, α, model; exact_Y=exact_Y, exposure_past=exposure_past
        ).prob

    c_mean = mean.(model_exp)
    g_mean = predict_mean_posterior(
        Y,
        X,
        α,
        model;
        exact_Y=exact_Y,
        exposure_past=exposure_past,
        exposure_future=exposure_future,
    )
    c_var = var.(model_exp)

    var_c_mean = fill(NaN, size(X)[1], size(model)[1])
    mean_c_var = fill(NaN, size(X)[1], size(model)[1])

    for i in 1:size(X)[1]
        var_c_mean[i, :] = (c_mean[:, :, i] .- g_mean[i, :]) .^ 2 * weights[i, :]
        mean_c_var[i, :] = c_var[:, :, i] * weights[i, :]
    end

    return var_c_mean + mean_c_var
end

"""
    predict_limit_prior(X, α, model, limit; 
        exposure_future = nothing)

Predicts the limit expected value (LEV) of response, 
given covariates `X`, 
logit regression coefficients `α` and a specified `model` of expert functions.

# Arguments
- `X`: A matrix of covariates.
- `α`: A matrix of logit regression coefficients.
- `model`: A matrix specifying the expert functions.
- `limit`: A matrix specifying the cutoff point.

# Optional Arguments
- `exposure_future`: A vector indicating the time exposure (future) of each observation. If nothing is supplied, it is set to 1.0 by default.

# Return Values
- A matrix of predicted limit expected value of response, based on prior probabilities.
"""
function predict_limit_prior(X, α, model, limit; exposure_future=nothing)
    if isnothing(exposure_future)
        exposure_future = fill(1.0, size(X)[1])
    end

    model_exp = exposurize_model(model; exposure=exposure_future)

    weights = predict_class_prior(X, α).prob

    result = fill(NaN, size(X)[1], size(model)[1])

    for i in 1:size(X)[1]
        means = vcat(
            [
                hcat(
                    [lev(model_exp[d, j, i], limit[i, d]) for j in 1:size(model_exp)[2]]...
                ) for d in 1:size(model_exp)[1]
            ]...,
        )
        result[i, :] = means * weights[i, :]
    end

    return result
end

"""
    predict_limit_posterior(Y, X, α, model, limit;
        exact_Y = true, exposure_past = nothing, exposure_future = nothing)

Predicts the limit expected value (LEV) of response, 
given observations `Y`, covariates `X`, 
logit regression coefficients `α` and a specified `model` of expert functions.

# Arguments
- `Y`: A matrix of responses.
- `X`: A matrix of covariates.
- `α`: A matrix of logit regression coefficients.
- `model`: A matrix specifying the expert functions.
- `limit`: A vector specifying the cutoff point.

# Optional Arguments
- `exact_Y`: `true` or `false` (default), indicating if `Y` is observed exactly or with censoring and truncation.
- `exposure_past`: A vector indicating the time exposure (past) of each observation. If nothing is supplied, it is set to 1.0 by default.
- `exposure_future`: A vector indicating the time exposure (future) of each observation. If nothing is supplied, it is set to 1.0 by default.

# Return Values
- A matrix of predicted limit expected value of response, based on posterior probabilities.
"""
function predict_limit_posterior(
    Y, X, α, model, limit; exact_Y=true, exposure_past=nothing, exposure_future=nothing
)
    if isnothing(exposure_past)
        exposure_past = fill(1.0, size(X)[1])
    end

    if isnothing(exposure_future)
        exposure_future = fill(1.0, size(X)[1])
    end

    model_exp = exposurize_model(model; exposure=exposure_future)

    weights =
        predict_class_posterior(
            Y, X, α, model; exact_Y=exact_Y, exposure_past=exposure_past
        ).prob

    result = fill(NaN, size(X)[1], size(model)[1])

    for i in 1:size(X)[1]
        means = vcat(
            [
                hcat(
                    [lev(model_exp[d, j, i], limit[i, d]) for j in 1:size(model_exp)[2]]...
                ) for d in 1:size(model_exp)[1]
            ]...,
        )
        result[i, :] = means * weights[i, :]
    end

    return result
end

"""
    predict_excess_prior(X, α, model, limit;
        exposure_future = nothing)

Predicts the excess expectation of response, 
given covariates `X`, 
logit regression coefficients `α` and a specified `model` of expert functions.

# Arguments
- `X`: A matrix of covariates.
- `α`: A matrix of logit regression coefficients.
- `model`: A matrix specifying the expert functions.
- `limit`: A vector specifying the cutoff point.

# Optional Arguments
- `exposure_future`: A vector indicating the time exposure (future) of each observation. If nothing is supplied, it is set to 1.0 by default.

# Return Values
- A matrix of predicted excess expectation of response, based on prior probabilities.
"""
function predict_excess_prior(X, α, model, limit; exposure_future=nothing)
    if isnothing(exposure_future)
        exposure_future = fill(1.0, size(X)[1])
    end

    model_exp = exposurize_model(model; exposure=exposure_future)

    weights = predict_class_prior(X, α).prob

    result = fill(NaN, size(X)[1], size(model)[1])

    for i in 1:size(X)[1]
        means = vcat(
            [
                hcat(
                    [
                        excess(model_exp[d, j, i], limit[i, d]) for
                        j in 1:size(model_exp)[2]
                    ]...,
                ) for d in 1:size(model_exp)[1]
            ]...,
        )
        result[i, :] = means * weights[i, :]
    end

    return result
end

"""
    predict_excess_posterior(Y, X, α, model, limit;
        exact_Y = true, exposure_past = nothing, exposure_future = nothing)

Predicts the excess expectation of response, 
given observations `Y`, covariates `X`, 
logit regression coefficients `α` and a specified `model` of expert functions.

# Arguments
- `Y`: A matrix of responses.
- `X`: A matrix of covariates.
- `α`: A matrix of logit regression coefficients.
- `model`: A matrix specifying the expert functions.
- `limit`: A vector specifying the cutoff point.

# Optional Arguments
- `exact_Y`: `true` or `false` (default), indicating if `Y` is observed exactly or with censoring and truncation.
- `exposure_past`: A vector indicating the time exposure (past) of each observation. If nothing is supplied, it is set to 1.0 by default.
- `exposure_future`: A vector indicating the time exposure (future) of each observation. If nothing is supplied, it is set to 1.0 by default.

# Return Values
- A matrix of predicted excess expectation of response, based on posterior probabilities.
"""
function predict_excess_posterior(
    Y, X, α, model, limit; exact_Y=true, exposure_past=nothing, exposure_future=nothing
)
    if isnothing(exposure_past)
        exposure_past = fill(1.0, size(X)[1])
    end

    if isnothing(exposure_future)
        exposure_future = fill(1.0, size(X)[1])
    end

    model_exp = exposurize_model(model; exposure=exposure_future)

    weights =
        predict_class_posterior(
            Y, X, α, model; exact_Y=exact_Y, exposure_past=exposure_past
        ).prob

    result = fill(NaN, size(X)[1], size(model)[1])

    for i in 1:size(X)[1]
        means = vcat(
            [
                hcat(
                    [
                        excess(model_exp[d, j, i], limit[i, d]) for
                        j in 1:size(model_exp)[2]
                    ]...,
                ) for d in 1:size(model_exp)[1]
            ]...,
        )
        result[i, :] = means * weights[i, :]
    end

    return result
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
            Roots.find_zero(
                y -> sum(weights .* exp.(expert_ll.(experts, 0.0, 0.0, y, Inf))) - p,
                (0.0, init_guess + 500),
            )
        catch
            NaN
        end
        return VaR
    end
end

function _calc_continuous_CTE(weights, experts, p, VaR)
    m = sum(vec(weights) .* vec(mean.(experts)))
    lim_ev = sum(vec(weights) .* vec(lev.(experts, VaR))) # [lev(model[k], VaR) for k in 1:length(model)]
    return VaR + (m - lim_ev) / (1 - p)
end

# calculate CTE based on VaR and p
# function _calc_continuous_mix_CTE(weights, experts, p, means, VaR)
#     return 
# end

"""
    predict_VaRCTE_prior(X, α, model, p;
        exposure_future = nothing)

Predicts the `p`-th value-at-risk (VaR) and conditional tail expectation (CTE) of response, 
given covariates `X`, 
logit regression coefficients `α` and a specified `model` of expert functions.

# Arguments
- `X`: A matrix of covariates.
- `α`: A matrix of logit regression coefficients.
- `model`: A matrix specifying the expert functions.
- `p`: A matrix of probabilities.

# Optional Arguments
- `exposure_future`: A vector indicating the time exposure (future) of each observation. If nothing is supplied, it is set to 1.0 by default.

# Return Values
- `VaR`: A matrix of predicted VaR of response, based on prior probabilities.
- `CTE`: A matrix of predicted CTE of response, based on prior probabilities.
"""
function predict_VaRCTE_prior(X, α, model, p; exposure_future=nothing)
    if isnothing(exposure_future)
        exposure_future = fill(1.0, size(X)[1])
    end

    model_exp = exposurize_model(model; exposure=exposure_future)

    weights = predict_class_prior(X, α).prob

    VaR = fill(NaN, size(X)[1], size(model)[1])
    CTE = fill(NaN, size(X)[1], size(model)[1])
    for i in 1:size(X)[1]
        for k in 1:size(model)[1]
            VaR[i, k] = _solve_continuous_mix_quantile(
                weights[i, :], model_exp[k, :, i], p[i, k]
            )
            CTE[i, k] = _calc_continuous_CTE(
                weights[i, :], model_exp[k, :, i], p[i, k], VaR[i, k]
            )
        end
    end
    # return VaR

    # VaR = vcat([hcat([_solve_continuous_mix_quantile(weights[i,:], model[k,:], p) for k in 1:size(model)[1] ]...) for i in 1:size(X)[1]]...)
    # CTE = vcat([hcat([_calc_continuous_CTE(weights[i,:], model[k,:], p, VaR[i,k]) for k in 1:size(model)[1] ]...) for i in 1:size(X)[1]]...)
    return (VaR=VaR, CTE=CTE)
end

"""
    predict_VaRCTE_posterior(Y, X, α, model, p;
        exact_Y = true, exposure_past = nothing, exposure_future = nothing)

Predicts the `p`-th value-at-risk (VaR) and conditional tail expectation (CTE) of response, 
given observations `Y`, covariates `X`,
logit regression coefficients `α` and a specified `model` of expert functions.

# Arguments
- `X`: A matrix of covariates.
- `α`: A matrix of logit regression coefficients.
- `model`: A matrix specifying the expert functions.
- `p`: A matrix of probabilities.

# Optional Arguments
- `exact_Y`: `true` or `false` (default), indicating if `Y` is observed exactly or with censoring and truncation.
- `exposure_past`: A vector indicating the time exposure (past) of each observation. If nothing is supplied, it is set to 1.0 by default.
- `exposure_future`: A vector indicating the time exposure (future) of each observation. If nothing is supplied, it is set to 1.0 by default.

# Return Values
- `VaR`: A matrix of predicted VaR of response, based on posterior probabilities.
- `CTE`: A matrix of predicted CTE of response, based on posterior probabilities.
"""
function predict_VaRCTE_posterior(
    Y, X, α, model, p; exact_Y=true, exposure_past=nothing, exposure_future=nothing
)
    if isnothing(exposure_past)
        exposure_past = fill(1.0, size(X)[1])
    end

    if isnothing(exposure_future)
        exposure_future = fill(1.0, size(X)[1])
    end

    model_exp = exposurize_model(model; exposure=exposure_future)

    weights =
        predict_class_posterior(
            Y, X, α, model; exact_Y=exact_Y, exposure_past=exposure_past
        ).prob

    VaR = fill(NaN, size(X)[1], size(model)[1])
    CTE = fill(NaN, size(X)[1], size(model)[1])
    for i in 1:size(X)[1]
        for k in 1:size(model)[1]
            VaR[i, k] = _solve_continuous_mix_quantile(
                weights[i, :], model_exp[k, :, i], p[i, k]
            )
            CTE[i, k] = _calc_continuous_CTE(
                weights[i, :], model_exp[k, :, i], p[i, k], VaR[i, k]
            )
        end
    end
    # return VaR

    # VaR = vcat([hcat([_solve_continuous_mix_quantile(weights[i,:], model[k,:], p) for k in 1:size(model)[1] ]...) for i in 1:size(X)[1]]...)
    # CTE = vcat([hcat([_calc_continuous_CTE(weights[i,:], model[k,:], p, VaR[i,k]) for k in 1:size(model)[1] ]...) for i in 1:size(X)[1]]...)
    return (VaR=VaR, CTE=CTE)
end
