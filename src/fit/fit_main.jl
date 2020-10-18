# This fitting function assumes 
function fit_main(Y, X, α_init, model;
                  penalty = true, pen_α = 5.0, pen_params = nothing,
                  ϵ = 1e-03, α_iter_max = 5.0, ecm_iter_max = 200,
                  grad_jump = true, grad_seq = nothing,
                  print_steps = true)

    # Make variables accessible with in the scope of `let`
    let α_em, gate_em, model_em, ll_em_list, ll_em, ll_em_np, ll_em_old, ll_em_np_old, iter
        # Initial loglik
        gate_init = LogitGating(α_init, X)
        ll_np_list = loglik_np(Y, gate_init, model)
        ll_init_np = ll_np_list.ll
        ll_penalty = penalty ? (pen_α(α_init) + penalty_params(model, pen_params)) : 0.0
        ll_init = ll_init_np + ll_penalty 

        if print_steps
            println("Initial loglik: $(ll_init_np) (no penalty), $(ll_init) (with penalty)")
        end

        # start em
        α_em = copy(α_init)
        model_em = copy(model)
        gate_em = LogitGating(α_em, X)
        ll_em_list = loglik_np(Y, gate_em, model_em)
        ll_em_np = ll_em_list.ll
        ll_em = ll_init
        ll_em_old = -Inf
        iter = 0

        while (ll_em - ll_em_old > ϵ) & (iter < ecm_iter_max)
            # Update counter and loglikelihood
            iter = iter + 1
            ll_em_np_old = ll_em_np
            ll_em_old = ll_em

            # E-Step
            z_e_obs = EM_E_z_obs(ll_em_list.gate_expert_ll_comp, ll_em_list.gate_expert_ll)
            z_e_lat = EM_E_z_lat(ll_em_list.gate_expert_tn_bar_comp, ll_em_list.gate_expert_tn_bar)
            k_e = EM_E_k(ll_em_list.gate_expert_tn)

            # println("$(ll_em_list.ll)")
            # println("$(z_e_obs)")
            # println("$(z_e_lat)")
            # println("$(k_e)")
        

            # M-Step: α
            α_em = EM_M_α(X, α_em, z_e_obs, z_e_lat, k_e, α_iter_max = α_iter_max, penalty = penalty, pen_α = pen_α)
            gate_em = LogitGating(α_em, X)
            ll_em_list = loglik_np(Y, gate_em, model_em)
            ll_em_np = ll_em_list.ll
            ll_em_penalty = penalty ? (pen_α(α_em) + penalty_params(model_em, pen_params)) : 0.0
            ll_em = ll_em_np + ll_em_penalty

            if print_steps
                println("Iteration $(iter), updating α:  $(ll_em_old) ->  $(ll_em)")
                # println(α_em)
            end
            # ll_em_old = ll_em


            α_em = α_em
            model_em = model_em
            gate_em = LogitGating(α_em, X)
            ll_em_list = loglik_np(Y, gate_em, model_em)
            ll_em_np = ll_em_list.ll
            ll_em_penalty = penalty ? (pen_α(α_em) + penalty_params(model_em, pen_params)) : 0.0
            ll_em = ll_em_np + ll_em_penalty
        end

        # α_em[1,1] = Inf
        # model_em[1,1] = LogNormalExpert(2, 3)

        # println("$(α_em), $(α_init), $(model), $(model_em), $(ll_em), $(ll_em_old)")
        return α_em
    end 
    

end


