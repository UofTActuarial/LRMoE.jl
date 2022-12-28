## Working tests

# using Random, LRMoE, Statistics, Clustering, Distributions, HypothesisTests
using LRMoE, Statistics, Clustering, Distributions, HypothesisTests
# Random.seed!(1234)


X = [1 2 3; 1 5 6; 1 99 3; 1 3 9; 2 5 7; 2 3 5]
Y = [1.0 2; 0 3; 2.5 3; 4.0 0; 0.0 0; 2.0 1]

n_comp = 3

μ = 2
σ = 1

X = rand(Uniform(-1, 1), 20000, 5)
α_true = rand(Uniform(-1, 1), 3, 5)
α_true[3, :] .= 0.0

model = [LRMoE.LogNormalExpert(μ, σ) LRMoE.LogNormalExpert(0.5*μ, 0.6*σ) LRMoE.LogNormalExpert(1.5*μ, 2.0*σ);
         LRMoE.ZILogNormalExpert(0.4, μ, σ) LRMoE.LogNormalExpert(0.5*μ, 0.6*σ) LRMoE.ZILogNormalExpert(0.80, 1.5*μ, 2.0*σ)]

model = [LRMoE.WeibullExpert(μ, σ) LRMoE.WeibullExpert(0.5*μ, 0.6*σ) LRMoE.WeibullExpert(1.5*μ, 2.0*σ);
         LRMoE.ZIWeibullExpert(0.4, μ, σ) LRMoE.WeibullExpert(0.5*μ, 0.6*σ) LRMoE.ZIWeibullExpert(0.80, 1.5*μ, 2.0*σ)]

model = [LRMoE.LogNormalExpert(μ, σ) LRMoE.ZILogNormalExpert(0.30, 0.5*μ, 0.6*σ) LRMoE.LogNormalExpert(1.5*μ, 2.0*σ);
         LRMoE.ZILogNormalExpert(0.4, μ, σ) LRMoE.LogNormalExpert(0.5*μ, 0.6*σ) LRMoE.ZILogNormalExpert(0.80, 1.5*μ, 2.0*σ)]

Y_sim = sim_dataset(α_true, X, model) 

tmp = cmm_init_exact(Y_sim, X, n_comp, ["continuous" "continuous"])

Y = hcat(fill(0, length(Y_sim[:,1])), Y_sim[:,1], Y_sim[:,1], fill(Inf, length(Y_sim[:,1])), fill(0, length(Y_sim[:,2])), 0.80.*Y_sim[:,2], 1.25.*Y_sim[:,2], fill(Inf, length(Y_sim[:,2])))

pen_params = [[[Inf 1.0 Inf], [Inf 1.0 Inf], [Inf 1.0 Inf]],
                      [[Inf 1.0 Inf], [Inf 1.0 Inf], [Inf 1.0 Inf]]]

result = fit_main(Y, X, tmp.α_init, tmp.ll_best, penalty = false, pen_params = pen_params)


Y

tmp = hcat([min.(Y[:, 4*(j-1)+4], 0.50 .* (Y[:, 4*(j-1)+2] + Y[:, 4*(j-1)+3])) for j in 1:Int((size(Y)[2]/4))]...)

Y_sim


tmp = cmm_init(Y, X, n_comp, ["continuous" "continuous"])


vcat([hcat([tmp.params_init[j][lb][rand(Distributions.Categorical(fill(1/length(tmp.params_init[1]), length(tmp.params_init[1]))), 1)] 
    for lb in 1:length(tmp.params_init[1])]...) 
    for j in 1:length(tmp.params_init)]...)

_sample_random_init(tmp.params_init)

# ttmp = tmp.ll_init[1]

# ttmp1 = findmax.(ttmp)

# ttmp1[1][2]

# tmp.ll_init[1]

# ttmp2 = vcat([hcat([tmp.params_init[j][lb][findmax.(tmp.ll_init[j])[lb][2]] for lb in unique(label)]...) for j in 1:size(Y)[2]]...)



# tmp1 = ExactOneSampleKSTest(Y_sim[:,2], LogNormal(μ, σ))

# pvalue(tmp1)
# show_params(tmp1)

# tmp2 = HypothesisTests.ksstats(Y_sim[:,2], LogNormal(μ, σ))


# tmp3 = ks_distance(Y_sim[:,2], LogNormalExpert(μ, σ))
# tmp4 = ks_distance(Y_sim[:,2], ZILogNormalExpert(0.9, μ, σ))

# tmp3 = ks_distance(Y_sim[:,1], LogNormalExpert(μ, σ))
# tmp4 = ks_distance(Y_sim[:,1], ZILogNormalExpert(0.90, μ, σ))



# sum(Y_sim[:,1] .== 0.0) / sum(Y_sim[:,1] .>= 0.0)
# sum(Y_sim[:,2] .== 0.0) / sum(Y_sim[:,2] .>= 0.0)