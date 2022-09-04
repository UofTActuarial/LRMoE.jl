function sim_components(model)
    return LRMoE.sim_expert.(model)
end

function sim_logit_gating(α, X)
    X = Array(X)
    probs = exp.(LogitGating(α, X))
    return hcat([rand(Distributions.Multinomial(1, probs[i, :])) for i in 1:size(X)[1]]...)'
end

function sim_dataset(α, X, model; exposure=nothing)
    X = Array(X)
    if isnothing(exposure)
        exposure = fill(1.0, size(X)[1])
    end
    model_expo = exposurize_model(model; exposure=exposure)
    gating_sim = sim_logit_gating(α, X)
    return vcat(
        [sim_components(model_expo[:, :, i]) * gating_sim[i, :] for i in 1:size(X)[1]]'...
    )
end
