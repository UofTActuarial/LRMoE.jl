function sim_components(model, sample_size)
    return [[hcat([sim_expert(model[j, k], sample_size) for k in 1:size(model)[2]]...) for j in 1:size(model)[1]]...]
end

function sim_logit_gating(α, X)
    probs = exp.(LogitGating(α, X))
    return hcat([rand(Distributions.Multinomial(1, probs[i,:])) for i in 1:size(X)[1]]...)'
end

function sim_dataset(α, X, model)
    dim_comp_sim = sim_components(model, size(X)[1])
    gating_sim = sim_logit_gating(α, X)
    return hcat([sum(gating_sim .* dim_comp_sim[j], dims = 2)  for j in 1:size(model)[1]]...)
end


