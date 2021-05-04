using Test
using Distributions
using StatsFuns
# using Random

μ = 1
σ = 2
# using LRMoE

# Random.seed!(1234)

# tmp = LRMoESTD([1 2 ; 3 4], ["a" "b"])
# tmp1 = LRMoESTDFit(tmp, true, 20, 1.1, 1.2, 1.3)
# summary(tmp1)

using Random, LRMoE, Statistics, Clustering, Distributions


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

@testset "fitting simulated: lognormal" begin

    # d = Distributions.Gamma(1.0, 2.0)

    # μμ = rand(d, 1)
    # σσ = rand(d, 1)

    # for (μ, σ) in zip(μμ, σσ)

        # X = rand(Uniform(-1, 1), 20000, 5)
        # α_true = rand(Uniform(-1, 1), 3, 5)
        # α_true[3, :] .= 0.0
        # model = [LRMoE.LogNormalExpert(μ, σ) LRMoE.LogNormalExpert(0.5*μ, 0.6*σ) LRMoE.LogNormalExpert(1.5*μ, 2.0*σ);
        #          LRMoE.ZILogNormalExpert(0.4, μ, σ) LRMoE.LogNormalExpert(0.5*μ, 0.6*σ) LRMoE.ZILogNormalExpert(0.80, 1.5*μ, 2.0*σ)]
        
        # pen_params = [[[Inf 1.0 Inf], [Inf 1.0 Inf], [Inf 1.0 Inf]],
        #               [[Inf 1.0 Inf], [Inf 1.0 Inf], [Inf 1.0 Inf]]]
        
        # Y_sim = sim_dataset(α_true, X, model)    
        # α_guess = copy(α_true)
        # α_guess .= 0.0
        # model_guess = [LRMoE.LogNormalExpert(0.8*μ, 1.2*σ) LRMoE.LogNormalExpert(μ, 0.9*σ) LRMoE.LogNormalExpert(1.0*μ, 2.5*σ);
        #                LRMoE.ZILogNormalExpert(0.50, 2.0*μ, 1.2*σ) LRMoE.LogNormalExpert(0.75*μ, 0.3*σ) LRMoE.ZILogNormalExpert(0.50, 1.75*μ, 1.0*σ)]

        # # Exact observation
        # Y = hcat(fill(0, length(Y_sim[:,1])), Y_sim[:,1], Y_sim[:,1], fill(Inf, length(Y_sim[:,1])), fill(0, length(Y_sim[:,2])), Y_sim[:,2], Y_sim[:,2], fill(Inf, length(Y_sim[:,2])))
        # result = fit_main(Y, X, α_guess, model_guess, penalty = false, pen_params = pen_params)
        # result = fit_LRMoE(Y, X, α_guess, model_guess, exact_Y = false, penalty = false, pen_α = 10.0, pen_params = pen_params)
        # result = fit_LRMoE(Y_sim, X, α_guess, model_guess, exact_Y = true, penalty = false, pen_α = 10.0, pen_params = pen_params)

        # result = fit_LRMoE(Y, X, α_guess, model_guess, exact_Y = false, penalty = false)
        # result = fit_LRMoE(Y, X, α_guess, model_guess, exact_Y = false, penalty = true)
        # result = fit_LRMoE(Y, X, α_guess, model_guess, exact_Y = false, penalty = true, pen_α = 20.0, pen_params = pen_params)
        # result = fit_LRMoE(Y_sim, X, α_guess, model_guess, exact_Y = true, penalty = true)
    
        # # With censoring
        # Y = hcat(fill(0, length(Y_sim[:,1])), Y_sim[:,1], Y_sim[:,1], fill(Inf, length(Y_sim[:,1])), fill(0, length(Y_sim[:,2])), 0.80.*Y_sim[:,2], 1.20.*Y_sim[:,2], fill(Inf, length(Y_sim[:,2])))
        # result = fit_main(Y, X, α_guess, model_guess, penalty = false, pen_params = pen_params)

        # Y = hcat(fill(0, length(Y_sim[:,1])), 0.75.*Y_sim[:,1], fill(Inf, length(Y_sim[:,1])), fill(Inf, length(Y_sim[:,1])), fill(0, length(Y_sim[:,2])), 0.80.*Y_sim[:,2], 1.20.*Y_sim[:,2], fill(Inf, length(Y_sim[:,2])))
        # result = fit_main(Y, X, α_guess, model_guess, penalty = false, pen_params = pen_params)

        # With truncation
        # Y = hcat(fill(0.0, length(Y_sim[:,1])), Y_sim[:,1], Y_sim[:,1], fill(Inf, length(Y_sim[:,1])), 0.30.*Y_sim[:,2], 0.80.*Y_sim[:,2], 1.20.*Y_sim[:,2], fill(Inf, length(Y_sim[:,2])))
        # Y = hcat( 0.80.*Y_sim[:,1], Y_sim[:,1], Y_sim[:,1], fill(Inf, length(Y_sim[:,1])), 0.30.*Y_sim[:,2], 0.80.*Y_sim[:,2], 1.20.*Y_sim[:,2], fill(Inf, length(Y_sim[:,2])))
        # Y = hcat(fill(0, length(Y_sim[:,1])), Y_sim[:,1], Y_sim[:,1], fill(Inf, length(Y_sim[:,1])), fill(0.0, length(Y_sim[:,2])), 0.80.*Y_sim[:,2], 1.20.*Y_sim[:,2], 2.50.*Y_sim[:,2])
        # result = fit_main(Y, X, α_guess, model_guess, penalty = false, pen_params = pen_params)

        # Y = hcat(fill(0, length(Y_sim[:,1])), Y_sim[:,1], Y_sim[:,1], 2.0 .*Y_sim[:,1], fill(0.0, length(Y_sim[:,2])), Y_sim[:,2], Y_sim[:,2], fill(Inf, length(Y_sim[:,2])))
        # result = fit_main(Y, X, α_guess, model_guess, penalty = false, pen_params = pen_params, ecm_iter_max = 20)


        # gate_tmp = LogitGating(α_guess, X)
        # ll_tmp = loglik_np(Y, gate_tmp, model_guess)
        # hcat(Y, EM_E_k(ll_tmp.gate_expert_tn_bar_k))

        # model = [LRMoE.ZILogNormalExpert(0.3, μ, σ) LRMoE.ZILogNormalExpert(0.2, 0.5*μ, 0.6*σ) LRMoE.ZILogNormalExpert(0.10, 1.5*μ, 2.0*σ);
        #          LRMoE.ZILogNormalExpert(0.4, μ, σ) LRMoE.ZILogNormalExpert(0.5, 0.5*μ, 0.6*σ) LRMoE.ZILogNormalExpert(0.80, 1.5*μ, 2.0*σ)]


        # model_guess = [LRMoE.ZILogNormalExpert(0.50, 0.8*μ, 1.2*σ) LRMoE.ZILogNormalExpert(0.50, μ, 0.9*σ) LRMoE.ZILogNormalExpert(0.50, 1.0*μ, 2.5*σ);
        #                LRMoE.ZILogNormalExpert(0.50, 2.0*μ, 1.2*σ) LRMoE.ZILogNormalExpert(0.50, 0.75*μ, 0.3*σ) LRMoE.ZILogNormalExpert(0.50, 1.75*μ, 1.0*σ)]
        
        # Y = hcat(fill(0, length(Y_sim[:,1])), Y_sim[:,1], Y_sim[:,1], 2.0 .*Y_sim[:,1], fill(0.0, length(Y_sim[:,2])), Y_sim[:,2], Y_sim[:,2], fill(Inf, length(Y_sim[:,2])))
        # result = fit_main(Y, X, α_guess, model_guess, penalty = false, pen_params = pen_params, ecm_iter_max = 20)


        # model_guess = [LRMoE.ZILogNormalExpert(0.50, 0.8*μ, 1.2*σ) LRMoE.ZILogNormalExpert(0.50, μ, 0.9*σ) LRMoE.ZILogNormalExpert(0.50, 1.0*μ, 2.5*σ);
        #                LRMoE.ZILogNormalExpert(0.50, 2.0*μ, 1.2*σ) LRMoE.ZILogNormalExpert(0.50, 0.75*μ, 0.3*σ) LRMoE.ZILogNormalExpert(0.50, 1.75*μ, 1.0*σ)]
        
        # Y = hcat(fill(0, length(Y_sim[:,1])), Y_sim[:,1], Y_sim[:,1], 2.0 .*Y_sim[:,1], fill(0.0, length(Y_sim[:,2])), Y_sim[:,2], Y_sim[:,2], fill(Inf, length(Y_sim[:,2])))
        # result = fit_main(Y, X, α_guess, model_guess, penalty = false, pen_params = pen_params, ecm_iter_max = 20)

        # gate_tmp = LogitGating(α_true, X)
        # ll_tmp = loglik_np(Y, gate_tmp, model_guess)

        # hcat(Y, expm1.( - ll_tmp.gate_expert_tn ), exp.(ll_tmp.gate_expert_tn_bar_comp) ./ exp.(ll_tmp.gate_expert_tn_bar))
        
        # hcat(Y, expm1.( - ll_tmp.gate_expert_tn ), EM_E_z_lat(ll_tmp.gate_expert_tn_bar_comp, ll_tmp.gate_expert_tn_bar))
        
        
    # end

end

@testset "fitting simulated: poisson" begin

    # d = Distributions.Poisson(10)

    # λλ = rand(d, 1)

    # for λ in λλ

        # X = rand(Uniform(-1, 1), 20000, 5)
        # α_true = rand(Uniform(-1, 1), 3, 5)
        # α_true[3, :] .= 0.0
        # model = [LRMoE.PoissonExpert(λ) LRMoE.PoissonExpert(0.5*λ) LRMoE.PoissonExpert(1.5*λ);
        #          LRMoE.ZIPoissonExpert(0.4, λ) LRMoE.PoissonExpert(1.2*λ) LRMoE.ZIPoissonExpert(0.80, 2.0*λ)]
        
        # pen_params = [[[2.0 1.0], [2.0 1.0], [2.0 1.0]],
        #               [[2.0 1.0], [2.0 1.0], [2.0 1.0]]]

        # pen_params = [[[2.0 10.0], [2.0 10.0], [2.0 10.0]],
        #               [[2.0 10.0], [2.0 10.0], [2.0 10.0]]]              
        
        # Y_sim = sim_dataset(α_true, X, model)    
        # α_guess = copy(α_true)
        # α_guess .= 0.0
        # model_guess = [LRMoE.PoissonExpert(0.8*λ) LRMoE.PoissonExpert(λ) LRMoE.PoissonExpert(1.0*λ);
        #                LRMoE.ZIPoissonExpert(0.50, 2.0*λ) LRMoE.PoissonExpert(0.75*λ) LRMoE.ZIPoissonExpert(0.50, 1.75*λ)]

        # # Exact observation
        # Y = hcat(fill(0, length(Y_sim[:,1])), Y_sim[:,1], Y_sim[:,1], fill(Inf, length(Y_sim[:,1])), fill(0, length(Y_sim[:,2])), Y_sim[:,2], Y_sim[:,2], fill(Inf, length(Y_sim[:,2])))
        # result = fit_main(Y, X, α_guess, model_guess, penalty = false, pen_params = pen_params)
        # result = fit_main(Y, X, α_guess, model_guess, penalty = true, pen_α = 10, pen_params = pen_params)
    
        # # With censoring
        # Y = hcat(fill(0, length(Y_sim[:,1])), Y_sim[:,1], Y_sim[:,1], fill(Inf, length(Y_sim[:,1])), fill(0, length(Y_sim[:,2])), floor.(0.80.*Y_sim[:,2]), ceil.(1.20.*Y_sim[:,2]), fill(Inf, length(Y_sim[:,2])))
        # result = fit_main(Y, X, α_guess, model_guess, penalty = false, pen_params = pen_params)
        # result = fit_main(Y, X, α_guess, model_guess, penalty = true, pen_α = 10, pen_params = pen_params)

        # Y = hcat(fill(0, length(Y_sim[:,1])), floor.(0.75.*Y_sim[:,1]), fill(Inf, length(Y_sim[:,1])), fill(Inf, length(Y_sim[:,1])), fill(0, length(Y_sim[:,2])), floor.(0.80.*Y_sim[:,2]), ceil.(1.20.*Y_sim[:,2]), fill(Inf, length(Y_sim[:,2])))
        # result = fit_main(Y, X, α_guess, model_guess, penalty = false, pen_params = pen_params)
        # result = fit_main(Y, X, α_guess, model_guess, penalty = true, pen_α = 10, pen_params = pen_params)

        # # With truncation
        # Y = hcat(fill(0.0, length(Y_sim[:,1])), Y_sim[:,1], Y_sim[:,1], fill(Inf, length(Y_sim[:,1])), floor.(0.30.*Y_sim[:,2]), floor.(0.80.*Y_sim[:,2]), ceil.(1.20.*Y_sim[:,2]), fill(Inf, length(Y_sim[:,2])))
        # result = fit_main(Y, X, α_guess, model_guess, penalty = false, pen_params = pen_params)
        # result = fit_main(Y, X, α_guess, model_guess, penalty = true, pen_α = 10, pen_params = pen_params)
        # Y = hcat( floor.(0.80.*Y_sim[:,1]), Y_sim[:,1], Y_sim[:,1], fill(Inf, length(Y_sim[:,1])), floor.(0.30.*Y_sim[:,2]), floor.(0.80.*Y_sim[:,2]), ceil.(1.20.*Y_sim[:,2]), fill(Inf, length(Y_sim[:,2])))
        # result = fit_main(Y, X, α_guess, model_guess, penalty = false, pen_params = pen_params)
        # result = fit_main(Y, X, α_guess, model_guess, penalty = true, pen_α = 10, pen_params = pen_params)

        # ###########################################################
        # #! THERE IS ERROR IN THE FIRST OF THE FOLLOWING THREE CASES
        # Y = hcat(fill(0, length(Y_sim[:,1])), Y_sim[:,1], Y_sim[:,1], fill(Inf, length(Y_sim[:,1])), fill(0.0, length(Y_sim[:,2])), floor.(0.80.*Y_sim[:,2]), ceil.(1.20.*Y_sim[:,2]), ceil.(2.50.*Y_sim[:,2]))
        # result = fit_main(Y, X, α_guess, model_guess, penalty = false, pen_params = pen_params)
        # result = fit_main(Y, X, α_guess, model_guess, penalty = true, pen_α = 10, pen_params = pen_params)

        # Y = hcat(fill(0, length(Y_sim[:,1])), Y_sim[:,1], Y_sim[:,1], fill(Inf, length(Y_sim[:,1])), fill(0.0, length(Y_sim[:,2])), floor.(0.80.*Y_sim[:,2]), ceil.(1.20.*Y_sim[:,2]), ceil.(2.50.*Y_sim[:,2]).+1)
        # result = fit_main(Y, X, α_guess, model_guess, penalty = false, pen_params = pen_params)
        # result = fit_main(Y, X, α_guess, model_guess, penalty = true, pen_α = 10, pen_params = pen_params)

        # Y = hcat(fill(0, length(Y_sim[:,1])), Y_sim[:,1], Y_sim[:,1], fill(Inf, length(Y_sim[:,1])), floor.(0.30.*Y_sim[:,2]), floor.(0.80.*Y_sim[:,2]), ceil.(1.20.*Y_sim[:,2]), ceil.(2.50.*Y_sim[:,2]))
        # result = fit_main(Y, X, α_guess, model_guess, penalty = false, pen_params = pen_params)
    

        
    # end

end

@testset "fitting simulated: binomial" begin

    # d = Distributions.Poisson(10)

    # λλ = rand(d, 1)

    # for λ in λλ

    #     n = 30
    #     p = 0.2

    #     X = rand(Uniform(-1, 1), 20000, 5)
    #     α_true = rand(Uniform(-1, 1), 3, 5)
    #     α_true[3, :] .= 0.0
    #     model = [LRMoE.BinomialExpert(n, p) LRMoE.BinomialExpert(n, 0.5*p) LRMoE.BinomialExpert(n, 1.3*p);
    #              LRMoE.ZIBinomialExpert(0.4, n, p) LRMoE.BinomialExpert(n, 1.2*p) LRMoE.ZIBinomialExpert(0.70, n, 2.0*p)]
        
    #     pen_params = [[[2.0 1.0], [2.0 1.0], [2.0 1.0]],
    #                   [[2.0 1.0], [2.0 1.0], [2.0 1.0]]]

    #     pen_params = [[[2.0 10.0], [2.0 10.0], [2.0 10.0]],
    #                   [[2.0 10.0], [2.0 10.0], [2.0 10.0]]]              
        
    #     Y_sim = sim_dataset(α_true, X, model)    
    #     α_guess = copy(α_true)
    #     α_guess .= 0.0

    #     model_guess = [LRMoE.BinomialExpert(n, 0.8*p) LRMoE.BinomialExpert(n, p) LRMoE.BinomialExpert(n, 1.0*p);
    #              LRMoE.ZIBinomialExpert(0.5, n, 2.0*p) LRMoE.BinomialExpert(n, 0.75*p) LRMoE.ZIBinomialExpert(0.50, n, 1.75*p)]


    #     # Exact observation
    #     Y = hcat(fill(0, length(Y_sim[:,1])), Y_sim[:,1], Y_sim[:,1], fill(Inf, length(Y_sim[:,1])), fill(0, length(Y_sim[:,2])), Y_sim[:,2], Y_sim[:,2], fill(Inf, length(Y_sim[:,2])))
    #     result = fit_main(Y, X, α_guess, model_guess, penalty = false, pen_params = pen_params)
    #     result = fit_main(Y, X, α_guess, model_guess, penalty = true, pen_α = 10, pen_params = pen_params)
    
    #     # With censoring
    #     Y = hcat(fill(0, length(Y_sim[:,1])), Y_sim[:,1], Y_sim[:,1], fill(Inf, length(Y_sim[:,1])), fill(0, length(Y_sim[:,2])), floor.(0.80.*Y_sim[:,2]), ceil.(1.20.*Y_sim[:,2]), fill(Inf, length(Y_sim[:,2])))
    #     result = fit_main(Y, X, α_guess, model_guess, penalty = false, pen_params = pen_params)
    #     result = fit_main(Y, X, α_guess, model_guess, penalty = true, pen_α = 10, pen_params = pen_params)

    #     Y = hcat(fill(0, length(Y_sim[:,1])), floor.(0.75.*Y_sim[:,1]), fill(Inf, length(Y_sim[:,1])), fill(Inf, length(Y_sim[:,1])), fill(0, length(Y_sim[:,2])), floor.(0.80.*Y_sim[:,2]), ceil.(1.20.*Y_sim[:,2]), fill(Inf, length(Y_sim[:,2])))
    #     result = fit_main(Y, X, α_guess, model_guess, penalty = false, pen_params = pen_params)
    #     result = fit_main(Y, X, α_guess, model_guess, penalty = true, pen_α = 10, pen_params = pen_params)

    #     # With truncation
    #     Y = hcat(fill(0.0, length(Y_sim[:,1])), Y_sim[:,1], Y_sim[:,1], fill(Inf, length(Y_sim[:,1])), floor.(0.30.*Y_sim[:,2]), floor.(0.80.*Y_sim[:,2]), ceil.(1.20.*Y_sim[:,2]), fill(Inf, length(Y_sim[:,2])))
    #     result = fit_main(Y, X, α_guess, model_guess, penalty = false, pen_params = pen_params)
    #     result = fit_main(Y, X, α_guess, model_guess, penalty = true, pen_α = 10, pen_params = pen_params)
    #     Y = hcat( floor.(0.80.*Y_sim[:,1]), Y_sim[:,1], Y_sim[:,1], fill(Inf, length(Y_sim[:,1])), floor.(0.30.*Y_sim[:,2]), floor.(0.80.*Y_sim[:,2]), ceil.(1.20.*Y_sim[:,2]), fill(Inf, length(Y_sim[:,2])))
    #     result = fit_main(Y, X, α_guess, model_guess, penalty = false, pen_params = pen_params)
    #     result = fit_main(Y, X, α_guess, model_guess, penalty = true, pen_α = 10, pen_params = pen_params)

        
    # end

end

@testset "fitting simulated: gamma" begin

    # d = Distributions.Gamma(1.0, 2.0)

    # μμ = rand(d, 1)
    # σσ = rand(d, 1)

    # for (μ, σ) in zip(μμ, σσ)

        # μ = 2
        # σ = 3

        # X = rand(Uniform(-1, 1), 20000, 5)
        # α_true = rand(Uniform(-1, 1), 3, 5)
        # α_true[3, :] .= 0.0
        # model = [LRMoE.GammaExpert(μ, σ) LRMoE.GammaExpert(0.5*μ, 0.6*σ) LRMoE.GammaExpert(1.5*μ, 2.0*σ);
        #          LRMoE.ZIGammaExpert(0.4, μ, σ) LRMoE.GammaExpert(0.5*μ, 0.6*σ) LRMoE.ZIGammaExpert(0.80, 1.5*μ, 2.0*σ)]
        
        # pen_params = [[[1.0 Inf 1.0 Inf], [1.0 Inf 1.0 Inf], [1.0 Inf 1.0 Inf]],
        #               [[1.0 Inf 1.0 Inf], [1.0 Inf 1.0 Inf], [1.0 Inf 1.0 Inf]]]
        
        # Y_sim = sim_dataset(α_true, X, model)    
        # α_guess = copy(α_true)
        # α_guess .= 0.0
        # model_guess = [LRMoE.GammaExpert(0.8*μ, 1.2*σ) LRMoE.GammaExpert(μ, 0.9*σ) LRMoE.GammaExpert(1.0*μ, 2.5*σ);
        #                LRMoE.ZIGammaExpert(0.50, 2.0*μ, 1.2*σ) LRMoE.GammaExpert(0.75*μ, 0.3*σ) LRMoE.ZIGammaExpert(0.50, 1.75*μ, 1.0*σ)]

    #     # Exact observation
    #     Y = hcat(fill(0, length(Y_sim[:,1])), Y_sim[:,1], Y_sim[:,1], fill(Inf, length(Y_sim[:,1])), fill(0, length(Y_sim[:,2])), Y_sim[:,2], Y_sim[:,2], fill(Inf, length(Y_sim[:,2])))
    #     result = fit_main(Y, X, α_guess, model_guess, penalty = false, pen_params = pen_params, ϵ = 0.01)    
    
    #     # With censoring
    #     Y = hcat(fill(0, length(Y_sim[:,1])), Y_sim[:,1], Y_sim[:,1], fill(Inf, length(Y_sim[:,1])), fill(0, length(Y_sim[:,2])), 0.80.*Y_sim[:,2], 1.20.*Y_sim[:,2], fill(Inf, length(Y_sim[:,2])))
    #     result = fit_main(Y, X, α_guess, model_guess, penalty = false, pen_params = pen_params, ϵ = 0.01)

    #     Y = hcat(fill(0, length(Y_sim[:,1])), 0.75.*Y_sim[:,1], fill(Inf, length(Y_sim[:,1])), fill(Inf, length(Y_sim[:,1])), fill(0, length(Y_sim[:,2])), 0.80.*Y_sim[:,2], 1.20.*Y_sim[:,2], fill(Inf, length(Y_sim[:,2])))
    #     result = fit_main(Y, X, α_guess, model_guess, penalty = false, pen_params = pen_params)

    #     # With truncation
    #     Y = hcat(fill(0.0, length(Y_sim[:,1])), Y_sim[:,1], Y_sim[:,1], fill(Inf, length(Y_sim[:,1])), 0.30.*Y_sim[:,2], 0.80.*Y_sim[:,2], 1.20.*Y_sim[:,2], fill(Inf, length(Y_sim[:,2])))
    #     Y = hcat( 0.80.*Y_sim[:,1], Y_sim[:,1], Y_sim[:,1], fill(Inf, length(Y_sim[:,1])), 0.30.*Y_sim[:,2], 0.80.*Y_sim[:,2], 1.20.*Y_sim[:,2], fill(Inf, length(Y_sim[:,2])))
    #     Y = hcat(fill(0, length(Y_sim[:,1])), Y_sim[:,1], Y_sim[:,1], fill(Inf, length(Y_sim[:,1])), fill(0.0, length(Y_sim[:,2])), 0.80.*Y_sim[:,2], 1.20.*Y_sim[:,2], 2.50.*Y_sim[:,2])
    #     result = fit_main(Y, X, α_guess, model_guess, penalty = false, pen_params = pen_params)

    #     Y = hcat(fill(0, length(Y_sim[:,1])), Y_sim[:,1], Y_sim[:,1], 2.0 .*Y_sim[:,1], fill(0.0, length(Y_sim[:,2])), Y_sim[:,2], Y_sim[:,2], fill(Inf, length(Y_sim[:,2])))
    #     result = fit_main(Y, X, α_guess, model_guess, penalty = false, pen_params = pen_params, ecm_iter_max = 20)


    #     gate_tmp = LogitGating(α_guess, X)
    #     ll_tmp = loglik_np(Y, gate_tmp, model_guess)
    #     hcat(Y, EM_E_k(ll_tmp.gate_expert_tn_bar_k))

        
    # end

end

@testset "fitting simulated: inversegaussian" begin

    # d = Distributions.Gamma(1.0, 2.0)

    # μμ = rand(d, 1)
    # σσ = rand(d, 1)

    # for (μ, σ) in zip(μμ, σσ)

    #     μ = 2
    #     σ = 3

    #     X = rand(Uniform(-1, 1), 20000, 5)
    #     α_true = rand(Uniform(-1, 1), 3, 5)
    #     α_true[3, :] .= 0.0
    #     model = [LRMoE.InverseGaussianExpert(μ, σ) LRMoE.InverseGaussianExpert(0.5*μ, 0.6*σ) LRMoE.InverseGaussianExpert(1.5*μ, 2.0*σ);
    #              LRMoE.ZIInverseGaussianExpert(0.4, μ, σ) LRMoE.InverseGaussianExpert(0.5*μ, 0.6*σ) LRMoE.ZIInverseGaussianExpert(0.80, 1.5*μ, 2.0*σ)]
        
    #     pen_params = [[[1.0 Inf 1.0 Inf], [1.0 Inf 1.0 Inf], [1.0 Inf 1.0 Inf]],
    #                   [[1.0 Inf 1.0 Inf], [1.0 Inf 1.0 Inf], [1.0 Inf 1.0 Inf]]]
        
    #     Y_sim = sim_dataset(α_true, X, model)    
    #     α_guess = copy(α_true)
    #     α_guess .= 0.0
    #     model_guess = [LRMoE.InverseGaussianExpert(0.8*μ, 1.2*σ) LRMoE.InverseGaussianExpert(μ, 0.9*σ) LRMoE.InverseGaussianExpert(1.0*μ, 2.5*σ);
    #                    LRMoE.ZIInverseGaussianExpert(0.50, 2.0*μ, 1.2*σ) LRMoE.InverseGaussianExpert(0.75*μ, 0.3*σ) LRMoE.ZIInverseGaussianExpert(0.50, 1.75*μ, 1.0*σ)]

    #     # Exact observation
    #     Y = hcat(fill(0, length(Y_sim[:,1])), Y_sim[:,1], Y_sim[:,1], fill(Inf, length(Y_sim[:,1])), fill(0, length(Y_sim[:,2])), Y_sim[:,2], Y_sim[:,2], fill(Inf, length(Y_sim[:,2])))
    #     result = fit_main(Y, X, α_guess, model_guess, penalty = false, pen_params = pen_params, ϵ = 0.01)    
    
    #     # With censoring
    #     Y = hcat(fill(0, length(Y_sim[:,1])), Y_sim[:,1], Y_sim[:,1], fill(Inf, length(Y_sim[:,1])), fill(0, length(Y_sim[:,2])), 0.80.*Y_sim[:,2], 1.20.*Y_sim[:,2], fill(Inf, length(Y_sim[:,2])))
    #     result = fit_main(Y, X, α_guess, model_guess, penalty = false, pen_params = pen_params, ϵ = 0.01)

    #     Y = hcat(fill(0, length(Y_sim[:,1])), 0.75.*Y_sim[:,1], fill(Inf, length(Y_sim[:,1])), fill(Inf, length(Y_sim[:,1])), fill(0, length(Y_sim[:,2])), 0.80.*Y_sim[:,2], 1.20.*Y_sim[:,2], fill(Inf, length(Y_sim[:,2])))
    #     result = fit_main(Y, X, α_guess, model_guess, penalty = false, pen_params = pen_params)

    #     # With truncation
    #     Y = hcat(fill(0.0, length(Y_sim[:,1])), Y_sim[:,1], Y_sim[:,1], fill(Inf, length(Y_sim[:,1])), 0.30.*Y_sim[:,2], 0.80.*Y_sim[:,2], 1.20.*Y_sim[:,2], fill(Inf, length(Y_sim[:,2])))
    #     Y = hcat( 0.80.*Y_sim[:,1], Y_sim[:,1], Y_sim[:,1], fill(Inf, length(Y_sim[:,1])), 0.30.*Y_sim[:,2], 0.80.*Y_sim[:,2], 1.20.*Y_sim[:,2], fill(Inf, length(Y_sim[:,2])))
    #     Y = hcat(fill(0, length(Y_sim[:,1])), Y_sim[:,1], Y_sim[:,1], fill(Inf, length(Y_sim[:,1])), fill(0.0, length(Y_sim[:,2])), 0.80.*Y_sim[:,2], 1.20.*Y_sim[:,2], 2.50.*Y_sim[:,2])
    #     result = fit_main(Y, X, α_guess, model_guess, penalty = false, pen_params = pen_params, ϵ = 0.01)

    #     Y = hcat(fill(0, length(Y_sim[:,1])), Y_sim[:,1], Y_sim[:,1], 2.0 .*Y_sim[:,1], fill(0.0, length(Y_sim[:,2])), Y_sim[:,2], Y_sim[:,2], fill(Inf, length(Y_sim[:,2])))
    #     result = fit_main(Y, X, α_guess, model_guess, penalty = false, pen_params = pen_params, ecm_iter_max = 20)


    #     gate_tmp = LogitGating(α_guess, X)
    #     ll_tmp = loglik_np(Y, gate_tmp, model_guess)
    #     hcat(Y, EM_E_k(ll_tmp.gate_expert_tn_bar_k))

        
    # end

end

@testset "fitting simulated: weibull" begin

    # d = Distributions.Gamma(1.0, 2.0)

    # μμ = rand(d, 1)
    # σσ = rand(d, 1)

    # for (μ, σ) in zip(μμ, σσ)

    #     μ = 2
    #     σ = 3

    #     X = rand(Uniform(-1, 1), 20000, 5)
    #     α_true = rand(Uniform(-1, 1), 3, 5)
    #     α_true[3, :] .= 0.0
    #     model = [LRMoE.WeibullExpert(μ, σ) LRMoE.WeibullExpert(0.5*μ, 0.6*σ) LRMoE.WeibullExpert(1.5*μ, 2.0*σ);
    #              LRMoE.ZIWeibullExpert(0.4, μ, σ) LRMoE.WeibullExpert(0.5*μ, 0.6*σ) LRMoE.ZIWeibullExpert(0.80, 1.5*μ, 2.0*σ)]
        
    #     pen_params = [[[1.0 Inf 1.0 Inf], [1.0 Inf 1.0 Inf], [1.0 Inf 1.0 Inf]],
    #                   [[1.0 Inf 1.0 Inf], [1.0 Inf 1.0 Inf], [1.0 Inf 1.0 Inf]]]
        
    #     Y_sim = sim_dataset(α_true, X, model)    
    #     α_guess = copy(α_true)
    #     α_guess .= 0.0
    #     model_guess = [LRMoE.WeibullExpert(0.8*μ, 1.2*σ) LRMoE.WeibullExpert(μ, 0.9*σ) LRMoE.WeibullExpert(1.0*μ, 2.5*σ);
    #                    LRMoE.ZIWeibullExpert(0.50, 2.0*μ, 1.2*σ) LRMoE.WeibullExpert(0.75*μ, 0.3*σ) LRMoE.ZIWeibullExpert(0.50, 1.75*μ, 1.0*σ)]
        
    #     # Exact observation
    #     Y = hcat(fill(0, length(Y_sim[:,1])), Y_sim[:,1], Y_sim[:,1], fill(Inf, length(Y_sim[:,1])), fill(0, length(Y_sim[:,2])), Y_sim[:,2], Y_sim[:,2], fill(Inf, length(Y_sim[:,2])))
    #     result = fit_main(Y, X, α_guess, model_guess, penalty = false, pen_params = pen_params, ϵ = 0.01)    
    
    #     # With censoring
    #     Y = hcat(fill(0, length(Y_sim[:,1])), Y_sim[:,1], Y_sim[:,1], fill(Inf, length(Y_sim[:,1])), fill(0, length(Y_sim[:,2])), 0.80.*Y_sim[:,2], 1.20.*Y_sim[:,2], fill(Inf, length(Y_sim[:,2])))
    #     result = fit_main(Y, X, α_guess, model_guess, penalty = false, pen_params = pen_params, ϵ = 0.01)

    #     Y = hcat(fill(0, length(Y_sim[:,1])), 0.75.*Y_sim[:,1], fill(Inf, length(Y_sim[:,1])), fill(Inf, length(Y_sim[:,1])), fill(0, length(Y_sim[:,2])), 0.80.*Y_sim[:,2], 1.20.*Y_sim[:,2], fill(Inf, length(Y_sim[:,2])))
    #     result = fit_main(Y, X, α_guess, model_guess, penalty = false, pen_params = pen_params)

    #     # With truncation
    #     Y = hcat(fill(0.0, length(Y_sim[:,1])), Y_sim[:,1], Y_sim[:,1], fill(Inf, length(Y_sim[:,1])), 0.30.*Y_sim[:,2], 0.80.*Y_sim[:,2], 1.20.*Y_sim[:,2], fill(Inf, length(Y_sim[:,2])))
    #     Y = hcat( 0.80.*Y_sim[:,1], Y_sim[:,1], Y_sim[:,1], fill(Inf, length(Y_sim[:,1])), 0.30.*Y_sim[:,2], 0.80.*Y_sim[:,2], 1.20.*Y_sim[:,2], fill(Inf, length(Y_sim[:,2])))
    #     Y = hcat(fill(0, length(Y_sim[:,1])), Y_sim[:,1], Y_sim[:,1], fill(Inf, length(Y_sim[:,1])), fill(0.0, length(Y_sim[:,2])), 0.80.*Y_sim[:,2], 1.20.*Y_sim[:,2], 2.50.*Y_sim[:,2])
    #     result = fit_main(Y, X, α_guess, model_guess, penalty = false, pen_params = pen_params, ϵ = 0.01)

    #     Y = hcat(fill(0, length(Y_sim[:,1])), Y_sim[:,1], Y_sim[:,1], 2.0 .*Y_sim[:,1], fill(0.0, length(Y_sim[:,2])), Y_sim[:,2], Y_sim[:,2], fill(Inf, length(Y_sim[:,2])))
    #     result = fit_main(Y, X, α_guess, model_guess, penalty = false, pen_params = pen_params, ecm_iter_max = 20)


    #     gate_tmp = LogitGating(α_guess, X)
    #     ll_tmp = loglik_np(Y, gate_tmp, model_guess)
    #     hcat(Y, EM_E_k(ll_tmp.gate_expert_tn_bar_k))

        
    # end

end

@testset "fitting simulated: lognormal, exact" begin

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

        # Exact observation
        Y = hcat(Y_sim[:,1], Y_sim[:,2])
        result = fit_exact(Y, X, α_guess, model_guess, penalty = false, pen_params = pen_params)
        
    end

end

@testset "fitting simulated: poisson, exact" begin

    # d = Distributions.Poisson(10)

    # λλ = rand(d, 1)

    # for λ in λλ

        X = rand(Uniform(-1, 1), 20000, 5)
        α_true = rand(Uniform(-1, 1), 3, 5)
        α_true[3, :] .= 0.0
        model = [LRMoE.PoissonExpert(λ) LRMoE.PoissonExpert(0.5*λ) LRMoE.PoissonExpert(1.5*λ);
                 LRMoE.ZIPoissonExpert(0.4, λ) LRMoE.PoissonExpert(1.2*λ) LRMoE.ZIPoissonExpert(0.80, 2.0*λ)]
        
        pen_params = [[[2.0 1.0], [2.0 1.0], [2.0 1.0]],
                      [[2.0 1.0], [2.0 1.0], [2.0 1.0]]]

        pen_params = [[[2.0 10.0], [2.0 10.0], [2.0 10.0]],
                      [[2.0 10.0], [2.0 10.0], [2.0 10.0]]]              
        
        Y_sim = sim_dataset(α_true, X, model)    
        α_guess = copy(α_true)
        α_guess .= 0.0
        model_guess = [LRMoE.PoissonExpert(0.8*λ) LRMoE.PoissonExpert(λ) LRMoE.PoissonExpert(1.0*λ);
                       LRMoE.ZIPoissonExpert(0.50, 2.0*λ) LRMoE.PoissonExpert(0.75*λ) LRMoE.ZIPoissonExpert(0.50, 1.75*λ)]

        # Exact observation
        Y = hcat(Y_sim[:,1], Y_sim[:,2])
        result = fit_exact(Y, X, α_guess, model_guess, penalty = false, pen_params = pen_params)
        result = fit_exact(Y, X, α_guess, model_guess, penalty = true, pen_α = 10, pen_params = pen_params)





    #     X = rand(Uniform(-1, 1), 20000, 5)
    #     α_true = rand(Uniform(-1, 1), 3, 5)
    #     α_true[3, :] .= 0.0
    #     model = [LRMoE.ZIPoissonExpert(0.4, λ) LRMoE.PoissonExpert(1.2*λ) LRMoE.ZIPoissonExpert(0.80, 2.0*λ)]
        
    #     pen_params = [[[2.0 1.0], [2.0 1.0], [2.0 1.0]]]
            
        
    #     Y_sim = sim_dataset(α_true, X, model)    
    #     α_guess = copy(α_true)
    #     α_guess .= 0.0
    #     model_guess = [LRMoE.ZIPoissonExpert(0.50, 2.0*λ) LRMoE.PoissonExpert(0.75*λ) LRMoE.ZIPoissonExpert(0.50, 1.75*λ)]

    #     # Exact observation
    #     Y = hcat(Y_sim[:,1], Y_sim[:,2])
    #     result = fit_exact(Y, X, α_guess, model_guess, penalty = false, pen_params = pen_params)
    #     result = fit_exact(Y, X, α_guess, model_guess, penalty = true, pen_α = 10, pen_params = pen_params)
        
    # end

end

@testset "fitting simulated: binomial, exact" begin

    # d = Distributions.Poisson(10)

    # λλ = rand(d, 1)

    # for λ in λλ

    #     n = 30
    #     p = 0.2

    #     X = rand(Uniform(-1, 1), 20000, 5)
    #     α_true = rand(Uniform(-1, 1), 3, 5)
    #     α_true[3, :] .= 0.0
        # model = [LRMoE.BinomialExpert(n, p) LRMoE.BinomialExpert(n, 0.5*p) LRMoE.BinomialExpert(n, 1.3*p);
        #          LRMoE.ZIBinomialExpert(0.4, n, p) LRMoE.BinomialExpert(n, 1.2*p) LRMoE.ZIBinomialExpert(0.70, n, 2.0*p)]
        
    #     pen_params = [[[2.0 1.0], [2.0 1.0], [2.0 1.0]],
    #                   [[2.0 1.0], [2.0 1.0], [2.0 1.0]]]

    #     pen_params = [[[2.0 10.0], [2.0 10.0], [2.0 10.0]],
    #                   [[2.0 10.0], [2.0 10.0], [2.0 10.0]]]              
        
    #     Y_sim = sim_dataset(α_true, X, model)    
    #     α_guess = copy(α_true)
    #     α_guess .= 0.0

    #     model_guess = [LRMoE.BinomialExpert(n, 0.8*p) LRMoE.BinomialExpert(n, p) LRMoE.BinomialExpert(n, 1.0*p);
    #              LRMoE.ZIBinomialExpert(0.5, n, 2.0*p) LRMoE.BinomialExpert(n, 0.75*p) LRMoE.ZIBinomialExpert(0.50, n, 1.75*p)]


    #     # Exact observation
    #     Y = hcat(Y_sim[:,1], Y_sim[:,2])
    #     result = fit_exact(Y, X, α_guess, model_guess, penalty = false, pen_params = pen_params)
    #     result = fit_exact(Y, X, α_guess, model_guess, penalty = true, pen_α = 10, pen_params = pen_params)


        # n = 30
        # p = 0.2

        # X = rand(Uniform(-1, 1), 20000, 5)
        # α_true = rand(Uniform(-1, 1), 2, 5)
        # α_true[2, :] .= 0.0
        # model = [LRMoE.BinomialExpert(n, p) LRMoE.BinomialExpert(n, 0.0);
        #          LRMoE.ZIBinomialExpert(0.4, n, p) LRMoE.ZIBinomialExpert(0.80, n, 2.0*p)]
        
        # pen_params = [[[2.0 1.0], [2.0 1.0], [2.0 1.0]],
        #               [[2.0 1.0], [2.0 1.0], [2.0 1.0]]]

        # pen_params = [[[2.0 10.0], [2.0 10.0], [2.0 10.0]],
        #               [[2.0 10.0], [2.0 10.0], [2.0 10.0]]]              
        
        # Y_sim = sim_dataset(α_true, X, model)    
        # α_guess = copy(α_true)
        # α_guess .= 0.0

        # model_guess = [LRMoE.BinomialExpert(n, 0.8*p) LRMoE.BinomialExpert(n, p);
        #          LRMoE.ZIBinomialExpert(0.5, n, 2.0*p) LRMoE.ZIBinomialExpert(0.50, n, 1.75*p)]


        # # Exact observation
        # Y = hcat(Y_sim[:,1], Y_sim[:,2])
        # result = fit_exact(Y, X, α_guess, model_guess, penalty = false, pen_params = pen_params)
        # result = fit_exact(Y, X, α_guess, model_guess, penalty = true, pen_α = 10, pen_params = pen_params)
        

        
    #     n = 30
    #     p = 0.2

    #     X = rand(Uniform(-1, 1), 20000, 5)
    #     α_true = rand(Uniform(-1, 1), 2, 5)
    #     α_true[2, :] .= 0.0
    #     # model = reshape([LRMoE.BinomialExpert(n, p) LRMoE.ZIBinomialExpert(0.30, n, 0.80*p)], 1, 2)
    #     model = reshape([LRMoE.BinomialExpert(n, p) LRMoE.ZIBinomialExpert(0.30, n, 0.80*p)], 1, 2)
        
    #     pen_params = [[[2.0 1.0], [2.0 1.0]]]           
        
    #     Y_sim = sim_dataset(α_true, X, model)    
    #     α_guess = copy(α_true)
    #     α_guess .= 0.0

    #     # model = reshape([LRMoE.BinomialExpert(n, 2*p) LRMoE.ZIBinomialExpert(0.50, n, 0.80*p)], 1, 2)
    #     model_guess = reshape([LRMoE.BinomialExpert(n, 2*p) LRMoE.ZIBinomialExpert(0.10, n, 0.80*p)], 1, 2)


    #     # Exact observation
    #     Y = hcat(Y_sim[:,1])
    #     result = fit_exact(Y, X, α_guess, model_guess, penalty = false, pen_params = pen_params)
    #     result = fit_exact(Y, X, α_guess, model_guess, penalty = true, pen_α = 10, pen_params = pen_params)

    #     gate_tmp = LogitGating(α_guess, X)
    #     ll_tmp = loglik_exact(Y, gate_tmp, model_guess)
        
    # end

end

@testset "fitting simulated: negativebinomial, exact" begin

    # d = Distributions.Poisson(10)

    # λλ = rand(d, 1)

    # for λ in λλ

        # n = 30
        # p = 0.2

        # X = rand(Uniform(-1, 1), 20000, 5)
        # α_true = rand(Uniform(-1, 1), 3, 5)
        # α_true[3, :] .= 0.0
        # model = [LRMoE.NegativeBinomialExpert(n, p) LRMoE.NegativeBinomialExpert(n, 0.2*p) LRMoE.NegativeBinomialExpert(n+5, 1.3*p);
        #          LRMoE.ZINegativeBinomialExpert(0.4, n, p) LRMoE.NegativeBinomialExpert(n+2, 1.2*p) LRMoE.ZINegativeBinomialExpert(0.70, n+5, 2.0*p)]
        
        # pen_params = [[[2.0 1.0], [2.0 1.0], [2.0 1.0]],
        #               [[2.0 1.0], [2.0 1.0], [2.0 1.0]]]

        # pen_params = [[[2.0 10.0], [2.0 10.0], [2.0 10.0]],
        #               [[2.0 10.0], [2.0 10.0], [2.0 10.0]]]              
        
        # Y_sim = sim_dataset(α_true, X, model)    
        # α_guess = copy(α_true)
        # α_guess .= 0.0

        # model_guess = [LRMoE.NegativeBinomialExpert(n, 0.8*p) LRMoE.NegativeBinomialExpert(n, p) LRMoE.NegativeBinomialExpert(n+2, 1.0*p);
        #          LRMoE.ZINegativeBinomialExpert(0.5, n, 2.0*p) LRMoE.NegativeBinomialExpert(n+1, 0.75*p) LRMoE.ZINegativeBinomialExpert(0.50, n+8, 1.75*p)]

        # model_guess = [LRMoE.NegativeBinomialExpert(n, 0.8*p) LRMoE.NegativeBinomialExpert(n, p) LRMoE.NegativeBinomialExpert(n+2, 1.0*p);
        #          LRMoE.ZINegativeBinomialExpert(0.5, n, 2.0*p) LRMoE.ZINegativeBinomialExpert(0.50, n+1, 0.75*p) LRMoE.ZINegativeBinomialExpert(0.50, n+8, 1.75*p)]         

        # # Exact observation
        # Y = hcat(Y_sim[:,1], Y_sim[:,2])
        # result = fit_exact(Y, X, α_guess, model_guess, penalty = false, pen_params = pen_params)
        # result = fit_exact(Y, X, α_guess, model_guess, penalty = true, pen_α = 10, pen_params = pen_params)


        # n = 10 # 30
        # p = 0.2

        # X = rand(Uniform(-1, 1), 20000, 5)
        # α_true = rand(Uniform(-1, 1), 2, 5)
        # α_true[2, :] .= 0.0
        # model = [LRMoE.NegativeBinomialExpert(n, p) LRMoE.NegativeBinomialExpert(n+5, 0.80*p);
        #          LRMoE.ZINegativeBinomialExpert(0.4, n, p) LRMoE.ZINegativeBinomialExpert(0.20, n+6, 2.0*p)]
        
        # pen_params = [[[2.0 1.0], [2.0 1.0]],
        #               [[2.0 1.0], [2.0 1.0]]]             
        
        # Y_sim = sim_dataset(α_true, X, model)    
        # α_guess = copy(α_true)
        # α_guess .= 0.0

        # model_guess = [LRMoE.NegativeBinomialExpert(n, 0.5*p) LRMoE.NegativeBinomialExpert(n, p);
        #          LRMoE.ZINegativeBinomialExpert(0.5, n, 2.0*p) LRMoE.ZINegativeBinomialExpert(0.50, n, 1.75*p)]


        # # Exact observation
        # Y = hcat(Y_sim[:,1], Y_sim[:,2])
        # Y = hcat(fill(0, length(Y_sim[:,1])), floor.(0.60 .* Y_sim[:,1]), ceil.(1.50.*Y_sim[:,1]), fill(Inf, length(Y_sim[:,1])))
        # result = fit_exact(Y, X, α_guess, model_guess, penalty = false, pen_params = pen_params)
        # result = fit_exact(Y, X, α_guess, model_guess, penalty = true, pen_α = 10, pen_params = pen_params)
        
        # result = fit_main(Y, X, α_guess, model_guess, penalty = false, pen_params = pen_params)


        # n = 10 # 30
        # p = 0.2

        # X = rand(Uniform(-1, 1), 20000, 5)
        # α_true = rand(Uniform(-1, 1), 2, 5)
        # α_true[2, :] .= 0.0
        # model = [LRMoE.NegativeBinomialExpert(2*n, p) LRMoE.NegativeBinomialExpert(n, 0.80*p);
        #          LRMoE.ZINegativeBinomialExpert(0.4, n, p) LRMoE.ZINegativeBinomialExpert(0.60, 3*n, 2.0*p)]
        
        # pen_params = [[[2.0 1.0], [2.0 1.0], [2.0 1.0]],
        #               [[2.0 1.0], [2.0 1.0], [2.0 1.0]]]

        # pen_params = [[[2.0 10.0], [2.0 10.0], [2.0 10.0]],
        #               [[2.0 10.0], [2.0 10.0], [2.0 10.0]]]              
        
        # Y_sim = sim_dataset(α_true, X, model)    
        # α_guess = copy(α_true)
        # α_guess .= 0.0

        # model_guess = [LRMoE.NegativeBinomialExpert(2*n, 0.5*p) LRMoE.NegativeBinomialExpert(n, p);
        #          LRMoE.ZINegativeBinomialExpert(0.5, n, 0.8*p) LRMoE.ZINegativeBinomialExpert(0.50, 3*n, 1.75*p)]


        # # Exact observation
        # Y = hcat(Y_sim[:,1], Y_sim[:,2])
        # result = fit_exact(Y, X, α_guess, model_guess, penalty = false, pen_params = pen_params)
        # result = fit_exact(Y, X, α_guess, model_guess, penalty = true, pen_α = 10, pen_params = pen_params)
        
        # Y = hcat(fill(0, length(Y_sim[:,1])), floor.(0.60 .* Y_sim[:,1]), ceil.(1.50.*Y_sim[:,1]), fill(Inf, length(Y_sim[:,1])))
        # result = fit_main(Y, X, α_guess, model_guess, penalty = false, pen_params = pen_params)
        
        # Y = hcat( floor.(0.20 .* Y_sim[:,1]), floor.(0.60 .* Y_sim[:,1]), ceil.(1.50.*Y_sim[:,1]), ceil.(5.0 .* Y_sim[:,1]))
        # result = fit_main(Y, X, α_guess, model_guess, penalty = false, pen_params = pen_params)
    # end

end

@testset "fitting simulated: gammacount, exact" begin

    # d = Distributions.Poisson(10)

    # λλ = rand(d, 1)

    # for λ in λλ

        # n = 10 # 30
        # p = 3

        # X = rand(Uniform(-1, 1), 20000, 5)
        # α_true = rand(Uniform(-1, 1), 2, 5)
        # α_true[2, :] .= 0.0
        # model = [LRMoE.GammaCountExpert(n, p) LRMoE.GammaCountExpert(n, 0.80*p);
        #          LRMoE.ZIGammaCountExpert(0.4, n, p) LRMoE.ZIGammaCountExpert(0.20, n, 2.0*p)]
        
        # pen_params = [[[1.0 Inf 1.0 Inf], [1.0 Inf 1.0 Inf]],
        #               [[1.0 Inf 1.0 Inf], [1.0 Inf 1.0 Inf]]]             
        
        # Y_sim = sim_dataset(α_true, X, model)    
        # α_guess = copy(α_true)
        # α_guess .= 0.0

        # model_guess = [LRMoE.GammaCountExpert(n, 0.5*p) LRMoE.GammaCountExpert(n, p);
        #          LRMoE.ZIGammaCountExpert(0.5, n, 2.0*p) LRMoE.ZIGammaCountExpert(0.50, n, 1.75*p)]


        # # Exact observation
        # Y = hcat(Y_sim[:,1], Y_sim[:,2])
        # Y = hcat(fill(0, length(Y_sim[:,1])), floor.(0.60 .* Y_sim[:,1]), ceil.(1.50.*Y_sim[:,1]), fill(Inf, length(Y_sim[:,1])))
        # result = fit_exact(Y, X, α_guess, model_guess, penalty = false, pen_params = pen_params)
        # result = fit_exact(Y, X, α_guess, model_guess, penalty = true, pen_α = 10, pen_params = pen_params)
        
        # result = fit_main(Y, X, α_guess, model_guess, penalty = false, pen_params = pen_params)



        # n = 10 # 30
        # p = 3

        # X = rand(Uniform(-1, 1), 20000, 5)
        # α_true = rand(Uniform(-1, 1), 2, 5)
        # α_true[2, :] .= 0.0
        # model = [LRMoE.GammaCountExpert(2*n, p) LRMoE.GammaCountExpert(n, 0.80*p);
        #          LRMoE.ZIGammaCountExpert(0.4, n, p) LRMoE.ZIGammaCountExpert(0.60, 3*n, 2.0*p)]
        
        # pen_params = [[[1.0 Inf 1.0 Inf], [1.0 Inf 1.0 Inf]],
        #               [[1.0 Inf 1.0 Inf], [1.0 Inf 1.0 Inf]]]          
        
        # Y_sim = sim_dataset(α_true, X, model)    
        # α_guess = copy(α_true)
        # α_guess .= 0.0

        # model_guess = [LRMoE.GammaCountExpert(2*n, 0.5*p) LRMoE.GammaCountExpert(n, p);
        #          LRMoE.ZIGammaCountExpert(0.5, n, 0.8*p) LRMoE.ZIGammaCountExpert(0.50, 3*n, 1.75*p)]


        # gate_tmp = LogitGating(α_guess, X)
        # ll_tmp = loglik_exact(Y, gate_tmp, model_guess)

        # ll_tmp = loglik_np(Y, gate_tmp, model_guess)

        # # Exact observation
        # Y = hcat(Y_sim[:,1], Y_sim[:,2])
        # result = fit_exact(Y, X, α_guess, model_guess, penalty = false, pen_params = pen_params)
        # result = fit_exact(Y, X, α_guess, model_guess, penalty = true, pen_α = 10, pen_params = pen_params)
        
        # Y = hcat(fill(0, length(Y_sim[:,1])), floor.(0.60 .* Y_sim[:,1]), ceil.(1.50.*Y_sim[:,1]), fill(Inf, length(Y_sim[:,1])))
        # result = fit_main(Y, X, α_guess, model_guess, penalty = false, pen_params = pen_params)
        
        # Y = hcat( floor.(0.20 .* Y_sim[:,1]), floor.(0.60 .* Y_sim[:,1]), ceil.(1.50.*Y_sim[:,1]), ceil.(5.0 .* Y_sim[:,1]))
        # result = fit_main(Y, X, α_guess, model_guess, penalty = false, pen_params = pen_params)
    # end

end

@testset "fitting simulated: gamma, exact" begin

    d = Distributions.Gamma(1.0, 2.0)

    μμ = rand(d, 1)
    σσ = rand(d, 1)

    for (μ, σ) in zip(μμ, σσ)

        X = rand(Uniform(-1, 1), 20000, 5)
        α_true = rand(Uniform(-1, 1), 3, 5)
        α_true[3, :] .= 0.0
        model = [LRMoE.GammaExpert(μ, σ) LRMoE.GammaExpert(0.5*μ, 0.6*σ) LRMoE.GammaExpert(1.5*μ, 2.0*σ);
                 LRMoE.ZIGammaExpert(0.4, μ, σ) LRMoE.GammaExpert(0.5*μ, 0.6*σ) LRMoE.ZIGammaExpert(0.80, 1.5*μ, 2.0*σ)]
        
        pen_params = [[[1.0 Inf 1.0 Inf], [1.0 Inf 1.0 Inf], [1.0 Inf 1.0 Inf]],
                      [[1.0 Inf 1.0 Inf], [1.0 Inf 1.0 Inf], [1.0 Inf 1.0 Inf]]]
        
        Y_sim = sim_dataset(α_true, X, model)    
        α_guess = copy(α_true)
        α_guess .= 0.0
        model_guess = [LRMoE.GammaExpert(0.8*μ, 1.2*σ) LRMoE.GammaExpert(μ, 0.9*σ) LRMoE.GammaExpert(1.0*μ, 2.5*σ);
                       LRMoE.ZIGammaExpert(0.50, 2.0*μ, 1.2*σ) LRMoE.GammaExpert(0.75*μ, 0.3*σ) LRMoE.ZIGammaExpert(0.50, 1.75*μ, 1.0*σ)]

        # Exact observation
        Y = hcat(Y_sim[:,1], Y_sim[:,2])
        result = fit_exact(Y, X, α_guess, model_guess, penalty = false, pen_params = pen_params)
        result = fit_exact(Y, X, α_guess, model_guess, penalty = true, pen_α = 10, pen_params = pen_params)
        
    end

end

@testset "fitting simulated: inversegaussian, exact" begin

    # d = Distributions.Gamma(1.0, 2.0)

    # μμ = rand(d, 1)
    # σσ = rand(d, 1)

    # for (μ, σ) in zip(μμ, σσ)

    #     X = rand(Uniform(-1, 1), 20000, 5)
    #     α_true = rand(Uniform(-1, 1), 3, 5)
    #     α_true[3, :] .= 0.0
    #     model = [LRMoE.InverseGaussianExpert(μ, σ) LRMoE.InverseGaussianExpert(0.5*μ, 0.6*σ) LRMoE.InverseGaussianExpert(1.5*μ, 2.0*σ);
    #              LRMoE.ZIInverseGaussianExpert(0.4, μ, σ) LRMoE.InverseGaussianExpert(0.5*μ, 0.6*σ) LRMoE.ZIInverseGaussianExpert(0.80, 1.5*μ, 2.0*σ)]
        
    #     pen_params = [[[1.0 Inf 1.0 Inf], [1.0 Inf 1.0 Inf], [1.0 Inf 1.0 Inf]],
    #                   [[1.0 Inf 1.0 Inf], [1.0 Inf 1.0 Inf], [1.0 Inf 1.0 Inf]]]
        
    #     Y_sim = sim_dataset(α_true, X, model)    
    #     α_guess = copy(α_true)
    #     α_guess .= 0.0
    #     model_guess = [LRMoE.InverseGaussianExpert(0.8*μ, 1.2*σ) LRMoE.InverseGaussianExpert(μ, 0.9*σ) LRMoE.InverseGaussianExpert(1.0*μ, 2.5*σ);
    #                    LRMoE.ZIInverseGaussianExpert(0.50, 2.0*μ, 1.2*σ) LRMoE.InverseGaussianExpert(0.75*μ, 0.3*σ) LRMoE.ZIInverseGaussianExpert(0.50, 1.75*μ, 1.0*σ)]

    #     # Exact observation
    #     Y = hcat(Y_sim[:,1], Y_sim[:,2])
    #     result = fit_exact(Y, X, α_guess, model_guess, penalty = false, pen_params = pen_params)
    #     result = fit_exact(Y, X, α_guess, model_guess, penalty = true, pen_α = 10, pen_params = pen_params)
        
    # end

end

@testset "fitting simulated: weibull, exact" begin

    # d = Distributions.Gamma(1.0, 2.0)

    # μμ = rand(d, 1)
    # σσ = rand(d, 1)

    # for (μ, σ) in zip(μμ, σσ)

    #     μ = 3   
    #     σ = 8
    #     X = rand(Uniform(-1, 1), 20000, 5)
    #     α_true = rand(Uniform(-1, 1), 3, 5)
    #     α_true[3, :] .= 0.0
    #     model = [LRMoE.WeibullExpert(μ, σ) LRMoE.WeibullExpert(0.5*μ, 0.6*σ) LRMoE.WeibullExpert(1.5*μ, 2.0*σ);
    #              LRMoE.ZIWeibullExpert(0.4, μ, σ) LRMoE.WeibullExpert(0.5*μ, 0.6*σ) LRMoE.ZIWeibullExpert(0.80, 1.5*μ, 2.0*σ)]
        
    #     pen_params = [[[1.0 Inf 1.0 Inf], [1.0 Inf 1.0 Inf], [1.0 Inf 1.0 Inf]],
    #                   [[1.0 Inf 1.0 Inf], [1.0 Inf 1.0 Inf], [1.0 Inf 1.0 Inf]]]
        
    #     Y_sim = sim_dataset(α_true, X, model)    
    #     α_guess = copy(α_true)
    #     α_guess .= 0.0
    #     model_guess = [LRMoE.WeibullExpert(0.8*μ, 1.2*σ) LRMoE.WeibullExpert(μ, 0.9*σ) LRMoE.WeibullExpert(1.0*μ, 2.5*σ);
    #                    LRMoE.ZIWeibullExpert(0.50, 2.0*μ, 1.2*σ) LRMoE.WeibullExpert(0.75*μ, 0.3*σ) LRMoE.ZIWeibullExpert(0.50, 1.75*μ, 1.0*σ)]

    #     # Exact observation
    #     Y = hcat(Y_sim[:,1], Y_sim[:,2])
    #     result = fit_exact(Y, X, α_guess, model_guess, penalty = false, pen_params = pen_params)
    #     result = fit_exact(Y, X, α_guess, model_guess, penalty = true, pen_α = 10, pen_params = pen_params)
        
    #     gate_tmp = LogitGating(α_guess, X)
    #     ll_tmp = loglik_np(Y, gate_tmp, model_guess)
    #     hcat(Y, ll_tmp.expert_ll_comp)

    #     ll_comp = ll_tmp.expert_ll_comp
    #     pos_idx = Y_sim[:,1] .!= 0.0

    # end

end

@testset "fitting simulated: burr, exact" begin

    # d = Distributions.Gamma(1.0, 2.0)

    # μμ = rand(d, 1)
    # σσ = rand(d, 1)

    # for (μ, σ) in zip(μμ, σσ)

    #     k = 3
    #     c = 5   
    #     λ = 10
    #     X = rand(Uniform(-1, 1), 20000, 5)
    #     α_true = rand(Uniform(-1, 1), 3, 5)
    #     α_true[3, :] .= 0.0
    #     model = [LRMoE.BurrExpert(k, c, λ) LRMoE.BurrExpert(0.5*k, 0.5*c, 2*λ) LRMoE.BurrExpert(1.5*k, 2*c, 3*λ);
    #              LRMoE.ZIBurrExpert(0.4, k, c, λ) LRMoE.BurrExpert(2*k, 0.5*c, 0.6*λ) LRMoE.ZIBurrExpert(0.80, k, 1.5*c, 2.0*λ)]
        
    #     pen_params = [[[1.0 Inf 1.0 Inf 1.0 Inf], [1.0 Inf 1.0 Inf 1.0 Inf], [1.0 Inf 1.0 Inf 1.0 Inf]],
    #                   [[1.0 Inf 1.0 Inf 1.0 Inf], [1.0 Inf 1.0 Inf 1.0 Inf], [1.0 Inf 1.0 Inf 1.0 Inf]]]

    #     pen_params = [[[2.0 1.0 2.0 1.0 2.0 1.0], [2.0 1.0 2.0 1.0 2.0 1.0], [2.0 1.0 2.0 1.0 2.0 1.0]],
    #                   [[2.0 1.0 2.0 1.0 2.0 1.0], [2.0 1.0 2.0 1.0 2.0 1.0], [2.0 1.0 2.0 1.0 2.0 1.0]]]
        
    #     Y_sim = sim_dataset(α_true, X, model)    
    #     α_guess = copy(α_true)
    #     α_guess .= 0.0
    #     model_guess = [LRMoE.BurrExpert(k, c, λ) LRMoE.BurrExpert(0.5*k, 0.5*c, 2*λ) LRMoE.BurrExpert(1.5*k, 2*c, 3*λ);
    #              LRMoE.ZIBurrExpert(0.4, k, 2*c, λ) LRMoE.BurrExpert(k, c, 0.6*λ) LRMoE.ZIBurrExpert(0.50, k, 2*c, 5*λ)]

    #     # Exact observation
    #     Y = hcat(Y_sim[:,1], Y_sim[:,2])
    #     result = fit_exact(Y, X, α_guess, model_guess, penalty = false, pen_params = pen_params)
    #     result = fit_exact(Y, X, α_guess, model_guess, penalty = true, pen_α = 10, pen_params = pen_params)
        
    #     gate_tmp = LogitGating(α_guess, X)
    #     ll_tmp = loglik_np(Y, gate_tmp, model_guess)
    #     hcat(Y, ll_tmp.expert_ll_comp)

    #     ll_comp = ll_tmp.expert_ll_comp
    #     pos_idx = Y_sim[:,1] .!= 0.0


    #     k = 3
    #     c = 5   
    #     λ = 10
    #     X = rand(Uniform(-1, 1), 20000, 5)
    #     α_true = rand(Uniform(-1, 1), 2, 5)
    #     α_true[2, :] .= 0.0
    #     model = [LRMoE.BurrExpert(k, c, λ) LRMoE.BurrExpert(0.5*k, 0.5*c, 2*λ)]

    #     pen_params = [[[1.0 Inf 1.0 Inf 1.0 Inf], [1.0 Inf 1.0 Inf 1.0 Inf], [1.0 Inf 1.0 Inf 1.0 Inf]],
    #                   [[1.0 Inf 1.0 Inf 1.0 Inf], [1.0 Inf 1.0 Inf 1.0 Inf], [1.0 Inf 1.0 Inf 1.0 Inf]]]
        
    #     pen_params = [[[2.0 1.0 2.0 1.0 2.0 1.0], [2.0 1.0 2.0 1.0 2.0 1.0], [2.0 1.0 2.0 1.0 2.0 1.0]],
    #                   [[2.0 1.0 2.0 1.0 2.0 1.0], [2.0 1.0 2.0 1.0 2.0 1.0], [2.0 1.0 2.0 1.0 2.0 1.0]]]
        
    #     Y_sim = sim_dataset(α_true, X, model)    
    #     α_guess = copy(α_true)
    #     α_guess .= 0.0
    #     model_guess = [LRMoE.BurrExpert(k, 2*c, 0.8*λ) LRMoE.BurrExpert(1.5*k, c, 3*λ)]

    #     # Exact observation
    #     Y = hcat(Y_sim[:,1])
    #     result = fit_exact(Y, X, α_guess, model_guess, penalty = false, pen_params = pen_params)
    #     result = fit_exact(Y, X, α_guess, model_guess, penalty = true, pen_α = 10, pen_params = pen_params)

    #     Y = hcat(fill(0.0, length(Y_sim[:,1])), 0.80 .* Y_sim[:,1], 1.20 .* Y_sim[:,1], fill(Inf, length(Y_sim[:,1])))
    #     result = fit_main(Y, X, α_guess, model_guess, penalty = true, pen_α = 10, pen_params = pen_params)
    # end

end