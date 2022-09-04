
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
    ZINegativeBinomialExpert()
    ZIBinomialExpert()
    ZIGammaCountExpert()
]

_default_expert_real = [
]

function cluster_covariates(X, n_cluster)
    X_std = (X .- mean(X; dims=1)) ./ sqrt.(var(X; dims=1))
    nan2num(X_std, 0.0) # Intercept: normalization produces NaN
    tmp = kmeans(Array(X_std'), n_cluster)
    return (groups=assignments(tmp), prop=counts(tmp) ./ size(X)[1])
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
    return hcat(
        [
            min.(
                Y[:, 4 * (j - 1) + 4],
                0.50 .* (Y[:, 4 * (j - 1) + 2] + Y[:, 4 * (j - 1) + 3]),
            ) for j in 1:Int((size(Y)[2] / 4))
        ]...,
    )
end

function _sample_random_init(params_init)
    return vcat(
        [
            hcat(
                [
                    params_init[j][lb][rand(
                        Distributions.Categorical(
                            fill(1 / length(params_init[j][lb]), length(params_init[j][lb]))
                        ),
                        1,
                    )]
                    for lb in 1:length(params_init[1])
                ]...,
            )
            for j in 1:length(params_init)
        ]...,
    )
end

function cmm_init_exact(Y, X, n_comp, type)
    n_dim = size(Y)[2]
    n_cov = size(X)[2]
    label, prop = cluster_covariates(X, n_comp)

    # initialize α: constant term according to prop, last class is reference
    α_init = fill(0.0, n_comp, n_cov)
    α_init[:, 1] = log.(prop) .- log(prop[n_comp])

    # summary statistics
    zero_y = [
        hcat(
            [
                sum(Y[label .== lb, j] .== 0.0) / length(Y[label .== lb, j]) for
                lb in unique(label)
            ]...,
        ) for j in 1:size(Y)[2]
    ]
    mean_y_pos = [
        hcat(
            [mean(Y[label .== lb, j][Y[label .== lb, j] .> 0.0]) for lb in unique(label)]...
        ) for j in 1:size(Y)[2]
    ]
    var_y_pos = [
        hcat(
            [var(Y[label .== lb, j][Y[label .== lb, j] .> 0.0]) for lb in unique(label)]...
        ) for j in 1:size(Y)[2]
    ]
    skewness_y_pos = [
        hcat(
            [
                skewness(Y[label .== lb, j][Y[label .== lb, j] .> 0.0]) for
                lb in unique(label)
            ]...,
        ) for j in 1:size(Y)[2]
    ]
    kurtosis_y_pos = [
        hcat(
            [
                kurtosis(Y[label .== lb, j][Y[label .== lb, j] .> 0.0]) for
                lb in unique(label)
            ]...,
        ) for j in 1:size(Y)[2]
    ]

    # initialize component distributions
    params_init = [
        [_params_init_switch(Y[label .== lb, j], type[j]) for lb in unique(label)] for
        j in 1:size(Y)[2]
    ]

    # evaluate loglikelihood
    ll_init = [
        [
            [
                sum(
                    expert_ll.(
                        params_init[j][lb][e],
                        fill(0.0, length(Y[:, j])),
                        Y[:, j],
                        Y[:, j],
                        fill(Inf, length(Y[:, j])),
                    ),
                )
                for e in 1:length(params_init[j][lb])
            ]
            for lb in unique(label)
        ]
        for j in 1:size(Y)[2]
    ]

    # highest ll_init model
    ll_best = vcat(
        [
            hcat(
                [params_init[j][lb][findmax.(ll_init[j])[lb][2]] for lb in unique(label)]...
            ) for j in 1:size(Y)[2]
        ]...,
    )

    # evaluate ks stat
    # ks_init = [[[ks_distance(Y[:,j], params_init[j][lb][e])
    #             for e in 1:length(params_init[j][lb])]
    #             for lb in unique(label)] 
    #             for j in 1:size(Y)[2]]

    # highest ks_init model
    # ks_best = vcat([hcat([params_init[j][lb][findmin.(ks_init[j])[lb][2]] for lb in unique(label)]...) for j in 1:size(Y)[2]]...)

    return (α_init=α_init, params_init=params_init, ll_init=ll_init, # ks_init = ks_init,
        ll_best=ll_best, # ks_best = ks_best,
        zero_y=zero_y,
        mean_y_pos=mean_y_pos, var_y_pos=var_y_pos, skewness_y_pos=skewness_y_pos,
        kurtosis_y_pos=kurtosis_y_pos)
end

"""
    cmm_init(Y, X, n_comp, type; exact_Y = false, n_random = 5)

Initialize an LRMoE model using the Clustered Method of Moments (CMM).

# Arguments
- `Y`: A matrix of response.
- `X`: A matrix of covariates.
- `n_comp`: Integer. Number of latent classes/components.
- `type`: A vector of either `continuous`, `discrete` or `real`, indicating the type of response by dimension.

# Optional Arguments
- `exact_Y`: `true` or `false` (default), indicating if `Y` is observed exactly or with censoring and truncation.
- `n_random`: Integer. Number of randomized initializations. 

# Return Values
- `zero_y`: Proportion of zeros in observed `Y`.
- `mean_y_pos`: Mean of positive observations in `Y`.
- `var_y_pos`: Variance of positive observations in `Y`.
- `skewness_y_pos`: Skewness of positive observations in `Y`.
- `kurtosis_y_pos`: Kurtosis of positive observations in `Y`.
- `α_init`: Initialization of logit regression coefficients `α`.
- `params_init`: Initializations of expert functions. It is a three-dimensional vector. 
    For example, `params_init[1][2]` initializes the 1st dimension of `Y` using the 2nd latent class,
    which is a vector of potential expert functions to choose from.
- `ll_init`: Calculates the loglikelihood of each expert function on the clustered groups of `Y`.
    For example, `ll_init[1][2][3]` is the loglikelihood of the 1st dimension of `Y`, calculated based
    on the 2nd latent classes and the 3rd initialized expert function in `params_init`.
- `ll_best`: An initialization chosen from `params_init` which yields the highest likelihood upon initialization.
- `random_init`: A list of `n_random` randomized initializations chosen from `params_init`.
"""
function cmm_init(Y, X, n_comp, type; exact_Y=false, n_random=5)
    Y_transform = exact_Y ? Y : _cmm_transform_inexact_Y(Y)
    tmp = cmm_init_exact(Y_transform, X, n_comp, type)

    if n_random >= 1
        random_init = [_sample_random_init(tmp.params_init) for i in 1:n_random]
    else
        random_init = nothing
    end

    return (zero_y=tmp.zero_y,
        mean_y_pos=tmp.mean_y_pos, var_y_pos=tmp.var_y_pos,
        skewness_y_pos=tmp.skewness_y_pos, kurtosis_y_pos=tmp.kurtosis_y_pos,
        α_init=tmp.α_init, params_init=tmp.params_init,
        ll_init=tmp.ll_init, # ks_init = tmp.ks_init,
        ll_best=tmp.ll_best, # ks_best = tmp.ks_best,
        random_init=random_init)
end
