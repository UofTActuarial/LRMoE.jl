using Test
using Distributions
using StatsFuns

@testset "fitting" begin

    d = Distributions.Gamma(1.0, 2.0)

    μμ = rand(d, 1)
    σσ = rand(d, 1)

    for (μ, σ) in zip(μμ, σσ)
        l = Distributions.LogNormal(μ, σ)
        y = rand(l, 200)

        # LogNormal
        r = LRMoE.LogNormalExpert(μ, σ)
        Y = hcat(fill(0, length(y)), y, y, fill(Inf, length(y)), fill(0, length(y)), 0.80.*y, 1.25.*y, fill(Inf, length(y)))
        model = [LRMoE.LogNormalExpert(μ, σ) LRMoE.LogNormalExpert(0.5*μ, 0.6*σ) LRMoE.LogNormalExpert(1.5*μ, 2.0*σ);
                LRMoE.ZILogNormalExpert(0.4, μ, σ) LRMoE.LogNormalExpert(0.5*μ, 0.6*σ) LRMoE.ZILogNormalExpert(0.80, 1.5*μ, 2.0*σ)]
        
        
        X = rand(Uniform(-1, 1), 200, 5)
        α_true = rand(Uniform(-1, 1), 3, 5)
        α_true[3, :] .= 0.0
    

        fit_main(Y, X, α_true, model, penalty = false)
    
    end

end