function fit_LRMoE(Y, X, α_init, model;
                    exact_Y = false,
                    penalty = true, pen_α = 1.0, pen_params = nothing,
                    ϵ = 1e-03, α_iter_max = 5, ecm_iter_max = 200,
                    grad_jump = true, grad_seq = nothing,
                    print_steps = true)
    
    # Convert possible dataframes to arrays
    # Y = Array(Y)
    # X = Array(X)

    if penalty == false
        pen_α = Inf
        pen_params = [LRMoE.no_penalty_init.(model[k,:]) for k in 1:size(model)[1]]
    elseif isnothing(pen_params)
        pen_params = [LRMoE.penalty_init.(model[k,:]) for k in 1:size(model)[1]]
    end

    if exact_Y == true
        tmp = fit_exact(Array(Y), Array(X), Array(α_init), model;
                        penalty = penalty, pen_α = pen_α, pen_params = pen_params,
                        ϵ = ϵ, α_iter_max = α_iter_max, ecm_iter_max = ecm_iter_max,
                        grad_jump = grad_jump, grad_seq = grad_seq,
                        print_steps = print_steps)
    else
        tmp = fit_main(Array(Y), Array(X), Array(α_init), model;
                        penalty = penalty, pen_α = pen_α, pen_params = pen_params,
                        ϵ = ϵ, α_iter_max = α_iter_max, ecm_iter_max = ecm_iter_max,
                        grad_jump = grad_jump, grad_seq = grad_seq,
                        print_steps = print_steps)
    end

    model_result = LRMoESTD(tmp.α_fit, tmp.model_fit)

    fit_result = LRMoESTDFit(model_result, tmp.converge, tmp.iter, tmp.ll, tmp.ll_np, tmp.AIC, tmp.BIC)
end