using Test
using Distributions
using StatsFuns

μ = 1
σ = 2
using LRMoE

@testset "fitting" begin

    # d = Distributions.Gamma(1.0, 2.0)

    # μμ = rand(d, 1)
    # σσ = rand(d, 1)

    # for (μ, σ) in zip(μμ, σσ)
    #     l = Distributions.LogNormal(μ, σ)
    #     y = rand(l, 200)

    #     # LogNormal
    #     r = LRMoE.LogNormalExpert(μ, σ)
    #     Y = hcat(fill(0, length(y)), y, y, fill(Inf, length(y)), fill(0, length(y)), 0.80.*y, 1.25.*y, fill(Inf, length(y)))
    
    #     model = [LRMoE.LogNormalExpert(μ, σ) LRMoE.LogNormalExpert(0.5*μ, 0.6*σ) LRMoE.LogNormalExpert(1.5*μ, 2.0*σ);
    #             LRMoE.LogNormalExpert(1.2*μ, σ) LRMoE.LogNormalExpert(0.8*μ, 0.6*σ) LRMoE.LogNormalExpert(1.5*μ, 0.5*σ)]
        
        
    #     pen_params = [[[Inf 1.0 Inf], [Inf 1.0 Inf], [Inf 1.0 Inf]],
    #             [[Inf 1.0 Inf], [Inf 1.0 Inf], [Inf 1.0 Inf]]]
        
    #     X = rand(Uniform(-1, 1), 200, 5)
    #     α_true = rand(Uniform(-1, 1), 3, 5)
    #     α_true[3, :] .= 0.0
    

    #     result = fit_main(Y, X, α_true, model, penalty = false, pen_params = pen_params)

    #     model = [LRMoE.LogNormalExpert(μ, σ) LRMoE.LogNormalExpert(0.5*μ, 0.6*σ) LRMoE.LogNormalExpert(1.5*μ, 2.0*σ);
    #             LRMoE.ZILogNormalExpert(0.4, μ, σ) LRMoE.LogNormalExpert(0.5*μ, 0.6*σ) LRMoE.ZILogNormalExpert(0.80, 1.5*μ, 2.0*σ)]

    #     result = fit_main(Y, X, α_true, model, penalty = false, pen_params = pen_params)
    
    # end

end

@testset "fitting simulated data" begin

    d = Distributions.Gamma(1.0, 2.0)

    μμ = rand(d, 1)
    σσ = rand(d, 1)

    for (μ, σ) in zip(μμ, σσ)

        X = rand(Uniform(-1, 1), 20000, 5)
        α_true = rand(Uniform(-1, 1), 3, 5)
        α_true[3, :] .= 0.0
        model = [LRMoE.LogNormalExpert(μ, σ) LRMoE.LogNormalExpert(0.5*μ, 0.6*σ) LRMoE.LogNormalExpert(1.5*μ, 2.0*σ);
                 LRMoE.ZILogNormalExpert(0.4, μ, σ) LRMoE.LogNormalExpert(0.5*μ, 0.6*σ) LRMoE.ZILogNormalExpert(0.80, 1.5*μ, 2.0*σ)]
        
        pen_params = [[[Inf 1.0 Inf], [Inf 1.0 Inf], [Inf 1.0 Inf]],
                      [[Inf 1.0 Inf], [Inf 1.0 Inf], [Inf 1.0 Inf]]]
        
        Y_sim = sim_dataset(α_true, X, model)    
        α_guess = copy(α_true)
        α_guess .= 0.0
        model_guess = [LRMoE.LogNormalExpert(0.8*μ, 1.2*σ) LRMoE.LogNormalExpert(μ, 0.9*σ) LRMoE.LogNormalExpert(1.0*μ, 2.5*σ);
                       LRMoE.ZILogNormalExpert(0.50, 2.0*μ, 1.2*σ) LRMoE.LogNormalExpert(0.75*μ, 0.3*σ) LRMoE.ZILogNormalExpert(0.50, 1.75*μ, 1.0*σ)]

        # # Exact observation
        # Y = hcat(fill(0, length(Y_sim[:,1])), Y_sim[:,1], Y_sim[:,1], fill(Inf, length(Y_sim[:,1])), fill(0, length(Y_sim[:,2])), Y_sim[:,2], Y_sim[:,2], fill(Inf, length(Y_sim[:,2])))
        # result = fit_main(Y, X, α_guess, model_guess, penalty = false, pen_params = pen_params)
    
        # # With censoring
        # Y = hcat(fill(0, length(Y_sim[:,1])), Y_sim[:,1], Y_sim[:,1], fill(Inf, length(Y_sim[:,1])), fill(0, length(Y_sim[:,2])), 0.80.*Y_sim[:,2], 1.20.*Y_sim[:,2], fill(Inf, length(Y_sim[:,2])))
        # result = fit_main(Y, X, α_guess, model_guess, penalty = false, pen_params = pen_params)

        # Y = hcat(fill(0, length(Y_sim[:,1])), 0.75.*Y_sim[:,1], fill(Inf, length(Y_sim[:,1])), fill(Inf, length(Y_sim[:,1])), fill(0, length(Y_sim[:,2])), 0.80.*Y_sim[:,2], 1.20.*Y_sim[:,2], fill(Inf, length(Y_sim[:,2])))
        # result = fit_main(Y, X, α_guess, model_guess, penalty = false, pen_params = pen_params)

        # With truncation
        Y = hcat(fill(0.0, length(Y_sim[:,1])), Y_sim[:,1], Y_sim[:,1], fill(Inf, length(Y_sim[:,1])), 0.30.*Y_sim[:,2], 0.80.*Y_sim[:,2], 1.20.*Y_sim[:,2], fill(Inf, length(Y_sim[:,2])))
        Y = hcat( 0.80.*Y_sim[:,1], Y_sim[:,1], Y_sim[:,1], fill(Inf, length(Y_sim[:,1])), 0.30.*Y_sim[:,2], 0.80.*Y_sim[:,2], 1.20.*Y_sim[:,2], fill(Inf, length(Y_sim[:,2])))
        # Y = hcat(fill(0, length(Y_sim[:,1])), Y_sim[:,1], Y_sim[:,1], fill(Inf, length(Y_sim[:,1])), fill(0.0, length(Y_sim[:,2])), 0.80.*Y_sim[:,2], 1.20.*Y_sim[:,2], 2.50.*Y_sim[:,2])
        result = fit_main(Y, X, α_guess, model_guess, penalty = false, pen_params = pen_params)

        # Y = hcat(fill(0, length(Y_sim[:,1])), Y_sim[:,1], Y_sim[:,1], 2.0 .*Y_sim[:,1], fill(0.0, length(Y_sim[:,2])), Y_sim[:,2], Y_sim[:,2], fill(Inf, length(Y_sim[:,2])))
        # result = fit_main(Y, X, α_guess, model_guess, penalty = false, pen_params = pen_params, ecm_iter_max = 20)



        # model_guess = [LRMoE.ZILogNormalExpert(0.50, 0.8*μ, 1.2*σ) LRMoE.ZILogNormalExpert(0.50, μ, 0.9*σ) LRMoE.ZILogNormalExpert(0.50, 1.0*μ, 2.5*σ);
        #                LRMoE.ZILogNormalExpert(0.50, 2.0*μ, 1.2*σ) LRMoE.ZILogNormalExpert(0.50, 0.75*μ, 0.3*σ) LRMoE.ZILogNormalExpert(0.50, 1.75*μ, 1.0*σ)]
        
        # Y = hcat(fill(0, length(Y_sim[:,1])), Y_sim[:,1], Y_sim[:,1], 2.0 .*Y_sim[:,1], fill(0.0, length(Y_sim[:,2])), Y_sim[:,2], Y_sim[:,2], fill(Inf, length(Y_sim[:,2])))
        # result = fit_main(Y, X, α_guess, model_guess, penalty = false, pen_params = pen_params, ecm_iter_max = 20)

        
        
    end

end