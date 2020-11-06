
_default_expert_continuous = [
    
    LogNormalExpert()
    GammaExpert()
    InverseGaussianExpert()

    ZILogNormalExpert()
    ZIGammaExpert()
    ZIInverseGaussianExpert()
]

_default_expert_discrete = [
    
    PoissonExpert()
    BinomialExpert()

    ZIPoissonExpert()
    ZIBinomialExpert()
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

function cmm_init_exact(Y, X, n_comp, type)
    
    n_dim = size(Y)[2]
    label, prop = cluster_covariates(X, n_comp)
    
    # initialize α: constant term according to prop, last class is reference
    α_init = fill(0.0, n_comp, n_dim)
    α_init[:,1] = log.(prop) .- log(prop[n_comp])

    # initialize component distributions
    params_init = [[_params_init_switch(Y[label.==lb,j], type[j]) for lb in unique(label)] for j in 1:size(Y)[2]]

    return params_init
end

function cmm_init(Y, X, n_comp; type = ["continuous"])
    
end

X = [1 2 3; 1 5 6; 1 99 3; 1 3 9; 2 5 7; 2 3 5]
Y = [1.0 2; 0 3; 2.5 3; 4.0 0; 0.0 0; 2.0 1]

n_comp = 2

# tmp1 = cluster_covariates(X, n_comp)

# rsl = cluster_covariates(X, 3)


tmp = cmm_init_exact(Y, X, 2, ["continuous" "discrete"])