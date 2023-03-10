"""
    fit_LRMoE(Y, X, α_init, model; ...)

Fit an LRMoE model.

# Arguments
- `Y`: A matrix of response.
- `X`: A matrix of covariates.
- `α`: A matrix of logit regression coefficients.
- `model`: A matrix specifying the expert functions.

# Optional Arguments
- `expusure`: an array of numerics, indicating the time invertal over which the count data (if applicable) are collected.
    If `nothing` is provided, it is set to 1.0 for all observations. It is assumed that all continuous expert functions are
    not affected by `exposure`.
- `exact_Y`: `true` or `false` (default), indicating if `Y` is observed exactly or with censoring and truncation.
- `penalty`: `true` (default) or `false`, indicating whether penalty is imposed on the magnitude of parameters.
- `pen_α`: a numeric penalty on the magnitude of logit regression coefficients. Default is 1.0.
- `pen_params`: an array of penalty term on the magnitude of parameters of component distributions/expert functions.
- `ϵ`: Stopping criterion on loglikelihood (stop when the increment is less than `ϵ`). Default is 0.001.
- `α_iter_max`: Maximum number of iterations when updating `α`. Default is 5.
- `ecm_iter_max`: Maximum number of iterations of the ECM algorithm. Default is 200.
- `grad_jump`: **IN DEVELOPMENT**
- `grad_seq`: **IN DEVELOPMENT**
- `print_steps`: `1` (default) or any integer, indicating whether intermediate updates of parameters should be logged (and how often).
    If `0`, no intermediate updates will be logged. Otherwise, it will be logged every `print_steps` iterations.

# Return Values
- `model_result.α_fit`: Fitted values of logit regression coefficients `α`.
- `model_result.comp_dist`: Fitted parameters of expert functions.
- `converge`: `true` or `false`, indicating whether the fitting procedure has converged.
- `iter`: Number of iterations passed in the fitting function.
- `ll`: Loglikelihood of the fitted model (with penalty on the magnitude of parameters).
- `ll_np`: Loglikelihood of the fitted model (without penalty on the magnitude of parameters).
- `AIC`: Akaike Information Criterion (AIC) of the fitted model.
- `BIC`: Bayesian Information Criterion (BIC) of the fitted model.
"""
function fit_LRMoE(Y, X, α_init, model;
    exposure=nothing,
    exact_Y=false,
    penalty=true, pen_α=1.0, pen_params=nothing,
    ϵ=1e-03, α_iter_max=5, ecm_iter_max=200,
    grad_jump=true, grad_seq=nothing,
    print_steps=1)

    # Convert possible dataframes to arrays
    # Y = Array(Y)
    # X = Array(X)

    if penalty == false
        pen_α = Inf
        pen_params = [LRMoE.no_penalty_init.(model[k, :]) for k in 1:size(model)[1]]
    elseif isnothing(pen_params)
        pen_params = [LRMoE.penalty_init.(model[k, :]) for k in 1:size(model)[1]]
    end

    if isnothing(exposure)
        exposure = fill(1.0, size(X)[1])
    end

    if !isinteger(print_steps) || (print_steps < 0)
        error("print_steps must be a nonnegative integer.")
    end

    if exact_Y == true
        tmp = fit_exact(Array(Y), Array(X), Array(α_init), model;
            exposure=exposure,
            penalty=penalty, pen_α=pen_α, pen_params=pen_params,
            ϵ=ϵ, α_iter_max=α_iter_max, ecm_iter_max=ecm_iter_max,
            grad_jump=grad_jump, grad_seq=grad_seq,
            print_steps=print_steps)
    else
        tmp = fit_main(Array(Y), Array(X), Array(α_init), model;
            exposure=exposure,
            penalty=penalty, pen_α=pen_α, pen_params=pen_params,
            ϵ=ϵ, α_iter_max=α_iter_max, ecm_iter_max=ecm_iter_max,
            grad_jump=grad_jump, grad_seq=grad_seq,
            print_steps=print_steps)
    end

    model_result = LRMoESTD(tmp.α_fit, tmp.model_fit)

    return fit_result = LRMoESTDFit(
        model_result, tmp.converge, tmp.iter, tmp.ll, tmp.ll_np, tmp.AIC, tmp.BIC
    )
end