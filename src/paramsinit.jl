
function cluster_covariates(X, n_cluster)
    # meanvec = mean(X, dims = 1)
    # sdvec = sqrt.(var(X, dims = 1))
    X_std = ( X .- mean(X, dims = 1) ) ./ sqrt.(var(X, dims = 1))
    nan2num(X_std, 0.0) # Intercept: normalization produces NaN
    tmp = kmeans(Array(X_std'), n_cluster)
    return (groups = assignments(tmp), prop = counts(tmp) ./ size(X)[1] )
end



function cmm_init_exact(Y, X, n_comp; type = ["continuous"])
    
    n_dim = size(Y)[2]
    label, prop = cluster_covariates(X, n_comp)
    
    # initialize α: constant term according to prop, last class is reference
    α_init = fill(0.0, n_comp, n_dim)
    α_init[:,1] = log.(prop) .- log(prop[n_comp])

    # initialize component distributions

    return α_init
end

function cmm_init(Y, X, n_comp; type = ["continuous"])
    
end

X = [1 2 3; 1 5 6; 1 99 3; 1 3 9]
Y = [1.0 2; 0 3; 2.5 3; 4.0 0]

n_comp = 2

tmp1 = cluster_covariates(X, n_comp)

rsl = cluster_covariates(X, 3)




cmm_init(1, X, 2)