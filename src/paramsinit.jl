
_default_expert_continuous = [
    
    LogNormalExpert()
    GammaExpert()
    InverseGaussianExpert()
    WeibullExpert()
    BurrExpert()

    ZILogNormalExpert()
    ZIGammaExpert()
    ZIInverseGaussianExpert()
    ZIWeibullExpert()
    ZIBurrExpert()
]

_default_expert_discrete = [
    
    PoissonExpert()
    NegativeBinomialExpert()
    BinomialExpert()
    GammaCountExpert()

    ZIPoissonExpert()
    NegativeBinomialExpert()
    ZIBinomialExpert()
    ZIGammaCountExpert()
]

_default_expert_real = [
    
    
]

function cluster_covariates(X, n_cluster)
    X_std = ( X .- mean(X, dims = 1) ) ./ sqrt.(var(X, dims = 1))
    nan2num(X_std, 0.0) # Intercept: normalization produces NaN
    tmp = kmeans(Array(X_std'), n_cluster)
    return (groups = assignments(tmp), prop = counts(tmp) ./ size(X)[1] )
end

function _params_init_switch(Y_j, type_j)
    if type_j == "continuous"
        return params_init.(Ref(Y_j), _default_expert_continuous)
    elseif type_j == "discrete"
        return params_init.(Ref(Y_j), _default_expert_discrete)
    elseif type_j == "real"
        return params_init.(Ref(Y_j), _default_expert_real)
    else
        error("Invalid specification of distribution types.")
    end
end

function _cmm_transform_inexact_Y(Y)
    return hcat([min.(Y[:, 4*(j-1)+4], 0.50 .* (Y[:, 4*(j-1)+2] + Y[:, 4*(j-1)+3])) for j in 1:Int((size(Y)[2]/4))]...)
end

function _sample_random_init(params_init)
    return vcat([hcat([params_init[j][lb][rand(Distributions.Categorical(fill(1/length(params_init[j][lb]), length(params_init[j][lb]))), 1)]
                for lb in 1:length(params_init[1])]...) 
                for j in 1:length(params_init)]...)
end

function cmm_init_exact(Y, X, n_comp, type)
    
    n_dim = size(Y)[2]
    n_cov = size(X)[2]
    label, prop = cluster_covariates(X, n_comp)
    
    # initialize α: constant term according to prop, last class is reference
    α_init = fill(0.0, n_comp, n_cov)
    α_init[:,1] = log.(prop) .- log(prop[n_comp])

    # initialize component distributions
    params_init = [[_params_init_switch(Y[label.==lb,j], type[j]) for lb in unique(label)] for j in 1:size(Y)[2]]

    # evaluate loglikelihood
    ll_init = [[[sum(expert_ll.(params_init[j][lb][e], fill(0.0, length(Y[:,j])), Y[:,j], Y[:,j], fill(Inf, length(Y[:,j]))))
                for e in 1:length(params_init[j][lb])]
                for lb in unique(label)] 
                for j in 1:size(Y)[2]]

    # highest ll_init model
    ll_best = vcat([hcat([params_init[j][lb][findmax.(ll_init[j])[lb][2]] for lb in unique(label)]...) for j in 1:size(Y)[2]]...)

    # evaluate ks stat
    ks_init = [[[ks_distance(Y[:,j], params_init[j][lb][e])
                for e in 1:length(params_init[j][lb])]
                for lb in unique(label)] 
                for j in 1:size(Y)[2]]

    # highest ks_init model
    ks_best = vcat([hcat([params_init[j][lb][findmin.(ks_init[j])[lb][2]] for lb in unique(label)]...) for j in 1:size(Y)[2]]...)

    return (α_init = α_init, params_init = params_init, ll_init = ll_init, ks_init = ks_init,
            ll_best = ll_best, ks_best = ks_best)
end

function cmm_init(Y, X, n_comp, type; exact_Y = false, n_random = 5)
    Y_transform = exact_Y ? Y : _cmm_transform_inexact_Y(Y)
    tmp = cmm_init_exact(Y_transform, X, n_comp, type)

    if n_random >= 1
        random_init = [_sample_random_init(tmp.params_init) for i in 1:n_random]
    else
        random_init = nothing
    end

    return (α_init = tmp.α_init, params_init = tmp.params_init, 
            ll_init = tmp.ll_init, ks_init = tmp.ks_init,
            ll_best = tmp.ll_best, ks_best = tmp.ks_best,
            random_init = random_init)
end


