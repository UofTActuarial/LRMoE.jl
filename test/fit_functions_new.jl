using Test
using Distributions
using StatsFuns

using LRMoE


@testset "Poisson, exact" begin

    # # Fit 20,000 observations as an example

    # param0 = rand(Distributions.Uniform(0, 1), 1)[1]
    # param1 = rand(Distributions.Uniform(0.5, 8), 1)[1]

    # X = rand(Distributions.Uniform(-5, 5), 20000, 7)
    # α = rand(Distributions.Uniform(-1, 1), 2, 7)
    # α[2,:] .= 0

    # model = [PoissonExpert(param1) PoissonExpert(2*param1);
    #         PoissonExpert(3*param1) PoissonExpert(0.5*param1)]

    # expos = rand(Distributions.Uniform(0.1, 5), 20000)
    # Y_sim = LRMoE.sim_dataset(α, X, model, exposure = expos)

    # α_guess = fill(0.0, 2, 7)
    # model_guess = [PoissonExpert(0.5*param1) PoissonExpert(3*param1);
    #                 PoissonExpert(2*param1) PoissonExpert(param1)]

    # pen_α = 5
    # pen_params = [[[1.0 Inf], [1.0 Inf]],
    #               [[1.0 Inf], [1.0 Inf]]]

    # result = LRMoE.fit_exact(Y_sim, X, α_guess, model_guess,
    #                     exposure = expos,
    #                     penalty = true, 
    #                     pen_α = pen_α, pen_params = pen_params,
    #                     # ϵ = 1e-03, α_iter_max = 5.0, ecm_iter_max = 200,
    #                     # grad_jump = true, grad_seq = nothing,
    #                     print_steps = true)
    
    # α
    # result.α_fit 
   
    # model
    # result.model_fit



end


@testset "ZIPoisson, exact" begin

    # # Fit 20,000 observations as an example

    # param0 = rand(Distributions.Uniform(0, 1), 4)
    # param1 = rand(Distributions.Uniform(0.5, 8), 1)[1]

    # X = rand(Distributions.Uniform(-5, 5), 20000, 7)
    # α = rand(Distributions.Uniform(-1, 1), 2, 7)
    # α[2,:] .= 0

    # model = [ZIPoissonExpert(param0[1], param1) ZIPoissonExpert(param0[2], 2*param1);
    #          ZIPoissonExpert(param0[3], 3*param1) ZIPoissonExpert(param0[4], 0.5*param1)]

    # expos = rand(Distributions.Uniform(0.1, 5), 20000)
    # Y_sim = LRMoE.sim_dataset(α, X, model, exposure = expos)

    # α_guess = fill(0.0, 2, 7)
    # model_guess = [ZIPoissonExpert(0.50, 0.5*param1) ZIPoissonExpert(0.50, 3*param1);
    #                ZIPoissonExpert(0.50, 2*param1) ZIPoissonExpert(0.50, param1)]

    # pen_α = 5
    # pen_params = [[[1.0 Inf], [1.0 Inf]],
    #               [[1.0 Inf], [1.0 Inf]]]

    # result = LRMoE.fit_exact(Y_sim, X, α_guess, model_guess,
    #                     exposure = expos,
    #                     penalty = true, 
    #                     pen_α = pen_α, pen_params = pen_params,
    #                     # ϵ = 1e-03, α_iter_max = 5.0, ecm_iter_max = 200,
    #                     # grad_jump = true, grad_seq = nothing,
    #                     print_steps = true)
    
    # α
    # result.α_fit 
   
    # model
    # result.model_fit
    
end


@testset "NegativeBinomial, exact" begin

    # Fit 50,000 observations as an example

    # param0 = rand(Distributions.Uniform(0, 1), 1)[1]
    # param1 = rand(Distributions.Uniform(0.5, 8), 4)
    # param2 = rand(Distributions.Uniform(0, 1), 4)

    # X = rand(Distributions.Uniform(-5, 5), 50000, 7)
    # α = rand(Distributions.Uniform(-1, 1), 2, 7)
    # α[2,:] .= 0

    # model = [NegativeBinomialExpert(param1[1], param2[1]) NegativeBinomialExpert(param1[2], param2[2]);
    #          NegativeBinomialExpert(param1[3], param2[3]) NegativeBinomialExpert(param1[4], param2[4])]

    # expos = rand(Distributions.Uniform(0.1, 5), 50000)
    # Y_sim = LRMoE.sim_dataset(α, X, model, exposure = expos)

    # α_guess = fill(0.0, 2, 7)
    # model_guess = [NegativeBinomialExpert(0.5*param1[1], 0.5) NegativeBinomialExpert(3*param1[2], 0.5);
    #                NegativeBinomialExpert(2*param1[3], 0.5) NegativeBinomialExpert(0.2*param1[4], 0.5)]

    # pen_α = 5
    # pen_params = [[[1.0 Inf], [1.0 Inf]],
    #               [[1.0 Inf], [1.0 Inf]]]

    # result = LRMoE.fit_exact(Y_sim, X, α_guess, model_guess,
    #                     exposure = expos,
    #                     penalty = true, 
    #                     pen_α = pen_α, pen_params = pen_params,
    #                     # ϵ = 1e-03, α_iter_max = 5.0, ecm_iter_max = 200,
    #                     # grad_jump = true, grad_seq = nothing,
    #                     print_steps = true)
    
    # α
    # result.α_fit 
   
    # model
    # result.model_fit



end

@testset "ZINegativeBinomial, exact" begin

    # Fit 50,000 observations as an example

    # param0 = [0.25 0.30 0.80 0.75] # rand(Distributions.Uniform(0, 1), 4)
    # param1 = [3 2.5 8 10] # rand(Distributions.Uniform(0.5, 8), 4)
    # param2 = [0.75 0.38 0.26 0.5]# rand(Distributions.Uniform(0, 1), 4)

    # X = rand(Distributions.Uniform(-5, 5), 50000, 7)
    # α = rand(Distributions.Uniform(-1, 1), 2, 7)
    # α[2,:] .= 0

    # model = [ZINegativeBinomialExpert(param0[1], param1[1], param2[1]) ZINegativeBinomialExpert(param0[2], param1[2], param2[2]);
    #          ZINegativeBinomialExpert(param0[3], param1[3], param2[3]) ZINegativeBinomialExpert(param0[4], param1[4], param2[4])]

    # expos = rand(Distributions.Uniform(0.1, 5), 50000)
    # Y_sim = LRMoE.sim_dataset(α, X, model, exposure = expos)

    # α_guess = fill(0.0, 2, 7)
    # model_guess = [ZINegativeBinomialExpert(0.5, 0.5*param1[1], 0.5) ZINegativeBinomialExpert(0.5, 3*param1[2], 0.5);
    #                ZINegativeBinomialExpert(0.5, 2*param1[3], 0.5) ZINegativeBinomialExpert(0.5, 0.2*param1[4], 0.5)]

    # pen_α = 5
    # pen_params = [[[1.0 Inf], [1.0 Inf]],
    #               [[1.0 Inf], [1.0 Inf]]]

    # result = LRMoE.fit_exact(Y_sim, X, α_guess, model_guess,
    #                     exposure = expos,
    #                     penalty = true, 
    #                     pen_α = pen_α, pen_params = pen_params,
    #                     # ϵ = 1e-03, α_iter_max = 5.0, ecm_iter_max = 200,
    #                     # grad_jump = true, grad_seq = nothing,
    #                     print_steps = true)
    
    # α
    # result.α_fit 
   
    # model
    # result.model_fit



end


@testset "Binomial, exact" begin

    # Fit 50,000 observations as an example

    # param0 = rand(Distributions.Uniform(0, 1), 1)[1]
    # param1 = rand(5:35, 4)
    # param2 = rand(Distributions.Uniform(0, 1), 4)

    # X = rand(Distributions.Uniform(-5, 5), 50000, 7)
    # α = rand(Distributions.Uniform(-1, 1), 2, 7)
    # α[2,:] .= 0

    # model = [BinomialExpert(param1[1], param2[1]) BinomialExpert(param1[2], param2[2]);
    #          BinomialExpert(param1[3], param2[3]) BinomialExpert(param1[4], param2[4])]

    # expos = rand(Distributions.Uniform(0.1, 5), 50000)
    # Y_sim = LRMoE.sim_dataset(α, X, model, exposure = expos)

    # α_guess = fill(0.0, 2, 7)
    # model_guess = [BinomialExpert(param1[1], 0.4) BinomialExpert(param1[2], 0.5);
    #                BinomialExpert(param1[3], 0.6) BinomialExpert(param1[4], 0.5)]

    # pen_α = 5
    # pen_params = [[[1.0 Inf], [1.0 Inf]],
    #               [[1.0 Inf], [1.0 Inf]]]

    # result = LRMoE.fit_exact(Y_sim, X, α_guess, model_guess,
    #                     exposure = expos,
    #                     penalty = true, 
    #                     pen_α = pen_α, pen_params = pen_params,
    #                     # ϵ = 1e-03, α_iter_max = 5.0, ecm_iter_max = 200,
    #                     # grad_jump = true, grad_seq = nothing,
    #                     print_steps = true)
    
    # α
    # result.α_fit 
   
    # model
    # result.model_fit



end


@testset "ZIBinomial, exact" begin

    # Fit 50,000 observations as an example

    # param0 = rand(Distributions.Uniform(0, 1), 4)
    # param1 = rand(5:35, 4)
    # param2 = rand(Distributions.Uniform(0, 1), 4)

    # X = rand(Distributions.Uniform(-5, 5), 50000, 7)
    # α = rand(Distributions.Uniform(-1, 1), 2, 7)
    # α[2,:] .= 0

    # model = [ZIBinomialExpert(param0[1], param1[1], param2[1]) ZIBinomialExpert(param0[2], param1[2], param2[2]);
    #          ZIBinomialExpert(param0[3], param1[3], param2[3]) ZIBinomialExpert(param0[4], param1[4], param2[4])]

    # expos = rand(Distributions.Uniform(0.1, 5), 50000)
    # Y_sim = LRMoE.sim_dataset(α, X, model, exposure = expos)

    # α_guess = fill(0.0, 2, 7)
    # model_guess = [ZIBinomialExpert(0.5, param1[1], 0.4) ZIBinomialExpert(0.5, param1[2], 0.5);
    #                ZIBinomialExpert(0.5, param1[3], 0.6) ZIBinomialExpert(0.5, param1[4], 0.5)]

    # pen_α = 5
    # pen_params = [[[1.0 Inf], [1.0 Inf]],
    #               [[1.0 Inf], [1.0 Inf]]]

    # result = LRMoE.fit_exact(Y_sim, X, α_guess, model_guess,
    #                     exposure = expos,
    #                     penalty = true, 
    #                     pen_α = pen_α, pen_params = pen_params,
    #                     # ϵ = 1e-03, α_iter_max = 5.0, ecm_iter_max = 200,
    #                     # grad_jump = true, grad_seq = nothing,
    #                     print_steps = true)
    
    # α
    # result.α_fit 
   
    # model
    # result.model_fit



end


@testset "GammaCount, exact" begin

    # Fit 50,000 observations as an example

    # param0 = rand(Distributions.Uniform(0, 1), 1)[1]
    # param1 = rand(Distributions.Uniform(0.5, 10), 4)
    # param2 = rand(Distributions.Uniform(0.5, 10), 4)

    # X = rand(Distributions.Uniform(-5, 5), 50000, 7)
    # α = rand(Distributions.Uniform(-1, 1), 2, 7)
    # α[2,:] .= 0

    # model = [GammaCountExpert(param1[1], param2[1]) GammaCountExpert(param1[2], param2[2]);
    #          GammaCountExpert(param1[3], param2[3]) GammaCountExpert(param1[4], param2[4])]

    # expos = rand(Distributions.Uniform(0.1, 5), 50000)
    # Y_sim = LRMoE.sim_dataset(α, X, model, exposure = expos)

    # α_guess = fill(0.0, 2, 7)
    # model_guess = [GammaCountExpert(0.5*param1[1], 2*param2[1]) GammaCountExpert(1.5*param1[2], 1.2*param2[2]);
    #                GammaCountExpert(param1[3], 0.8*param2[3]) GammaCountExpert(1.1*param1[4], 0.75*param2[4])]

    # pen_α = 5
    # pen_params = [[[1.0 Inf 1.0 Inf], [1.0 Inf 1.0 Inf]],
    #               [[1.0 Inf 1.0 Inf], [1.0 Inf 1.0 Inf]]]

    # result = LRMoE.fit_exact(Y_sim, X, α_guess, model_guess,
    #                     exposure = expos,
    #                     penalty = true, 
    #                     pen_α = pen_α, pen_params = pen_params,
    #                     # ϵ = 1e-03, α_iter_max = 5.0, ecm_iter_max = 200,
    #                     # grad_jump = true, grad_seq = nothing,
    #                     print_steps = true)
    
    # α
    # result.α_fit 
   
    # model
    # result.model_fit



end


@testset "ZIGammaCount, exact" begin

    # Fit 50,000 observations as an example

    # param0 = rand(Distributions.Uniform(0, 1), 4)
    # param1 = rand(Distributions.Uniform(0.5, 10), 4)
    # param2 = rand(Distributions.Uniform(0.5, 10), 4)

    # X = rand(Distributions.Uniform(-5, 5), 50000, 7)
    # α = rand(Distributions.Uniform(-1, 1), 2, 7)
    # α[2,:] .= 0

    # model = [ZIGammaCountExpert(param0[1], param1[1], param2[1]) ZIGammaCountExpert(param0[2], param1[2], param2[2]);
    #          ZIGammaCountExpert(param0[3], param1[3], param2[3]) ZIGammaCountExpert(param0[4], param1[4], param2[4])]

    # expos = rand(Distributions.Uniform(0.1, 5), 50000)
    # Y_sim = LRMoE.sim_dataset(α, X, model, exposure = expos)

    # α_guess = fill(0.0, 2, 7)
    # model_guess = [ZIGammaCountExpert(0.5, 0.5*param1[1], 2*param2[1]) ZIGammaCountExpert(0.5, 1.5*param1[2], 1.2*param2[2]);
    #                ZIGammaCountExpert(0.5, param1[3], 0.8*param2[3]) ZIGammaCountExpert(0.5, 1.1*param1[4], 0.75*param2[4])]

    # pen_α = 5
    # pen_params = [[[1.0 Inf 1.0 Inf], [1.0 Inf 1.0 Inf]],
    #               [[1.0 Inf 1.0 Inf], [1.0 Inf 1.0 Inf]]]

    # result = LRMoE.fit_exact(Y_sim, X, α_guess, model_guess,
    #                     exposure = expos,
    #                     penalty = true, 
    #                     pen_α = pen_α, pen_params = pen_params,
    #                     # ϵ = 1e-03, α_iter_max = 5.0, ecm_iter_max = 200,
    #                     # grad_jump = true, grad_seq = nothing,
    #                     print_steps = true)
    
    # α
    # result.α_fit 
   
    # model
    # result.model_fit



end


@testset "LogNormal, exact" begin

    # Fit 50,000 observations as an example

    # param0 = rand(Distributions.Uniform(0, 1), 1)[1]
    # param1 = rand(Distributions.Uniform(0.5, 5), 4)
    # param2 = rand(Distributions.Uniform(0.5, 3), 4)

    # X = rand(Distributions.Uniform(-5, 5), 50000, 7)
    # α = rand(Distributions.Uniform(-1, 1), 2, 7)
    # α[2,:] .= 0

    # model = [LogNormalExpert(param1[1], param2[1]) LogNormalExpert(param1[2], param2[2]);
    #          LogNormalExpert(param1[3], param2[3]) LogNormalExpert(param1[4], param2[4])]

    # expos = rand(Distributions.Uniform(0.1, 5), 50000)
    # Y_sim = LRMoE.sim_dataset(α, X, model, exposure = expos)

    # α_guess = fill(0.0, 2, 7)
    # model_guess = [LogNormalExpert(0.5*param1[1], 1.2*param2[1]) LogNormalExpert(0.75*param1[2], 0.8*param2[2]);
    #                LogNormalExpert(2*param1[3], 1.25*param2[3]) LogNormalExpert(0.1*param1[4], param2[4])]

    # pen_α = 5
    # pen_params = [[[Inf 1.0 Inf], [Inf 1.0 Inf]],
    #               [[Inf 1.0 Inf], [Inf 1.0 Inf]]]

    # result = LRMoE.fit_exact(Y_sim, X, α_guess, model_guess,
    #                     exposure = expos,
    #                     penalty = true, 
    #                     pen_α = pen_α, pen_params = pen_params,
    #                     # ϵ = 1e-03, α_iter_max = 5.0, ecm_iter_max = 200,
    #                     # grad_jump = true, grad_seq = nothing,
    #                     print_steps = true)
    
    # α
    # result.α_fit 
   
    # model
    # result.model_fit



end

@testset "ZILogNormal, exact" begin

    # Fit 50,000 observations as an example

    # param0 = rand(Distributions.Uniform(0, 1), 4)
    # param1 = rand(Distributions.Uniform(0.5, 5), 4)
    # param2 = rand(Distributions.Uniform(0.5, 3), 4)

    # X = rand(Distributions.Uniform(-5, 5), 50000, 7)
    # α = rand(Distributions.Uniform(-1, 1), 2, 7)
    # α[2,:] .= 0

    # model = [ZILogNormalExpert(param0[1], param1[1], param2[1]) ZILogNormalExpert(param0[2], param1[2], param2[2]);
    #          ZILogNormalExpert(param0[3], param1[3], param2[3]) ZILogNormalExpert(param0[4], param1[4], param2[4])]

    # expos = rand(Distributions.Uniform(0.1, 5), 50000)
    # Y_sim = LRMoE.sim_dataset(α, X, model, exposure = expos)

    # α_guess = fill(0.0, 2, 7)
    # model_guess = [ZILogNormalExpert(0.5, 0.5*param1[1], 1.2*param2[1]) ZILogNormalExpert(0.5, 0.75*param1[2], 0.8*param2[2]);
    #                ZILogNormalExpert(0.5, 2*param1[3], 1.25*param2[3]) ZILogNormalExpert(0.5, 0.1*param1[4], param2[4])]

    # pen_α = 5
    # pen_params = [[[Inf 1.0 Inf], [Inf 1.0 Inf]],
    #               [[Inf 1.0 Inf], [Inf 1.0 Inf]]]

    # result = LRMoE.fit_exact(Y_sim, X, α_guess, model_guess,
    #                     exposure = expos,
    #                     penalty = true, 
    #                     pen_α = pen_α, pen_params = pen_params,
    #                     # ϵ = 1e-03, α_iter_max = 5.0, ecm_iter_max = 200,
    #                     # grad_jump = true, grad_seq = nothing,
    #                     print_steps = true)
    
    # α
    # result.α_fit 
   
    # model
    # result.model_fit



end


@testset "Gamma, exact" begin

    # Fit 50,000 observations as an example

    # param0 = rand(Distributions.Uniform(0, 1), 1)[1]
    # param1 = rand(Distributions.Uniform(3, 8), 4)
    # param2 = rand(Distributions.Uniform(0.5, 15), 4)

    # X = rand(Distributions.Uniform(-5, 5), 50000, 7)
    # α = rand(Distributions.Uniform(-1, 1), 2, 7)
    # α[2,:] .= 0

    # model = [GammaExpert(param1[1], param2[1]) GammaExpert(param1[2], param2[2]);
    #          GammaExpert(param1[3], param2[3]) GammaExpert(param1[4], param2[4])]

    # expos = rand(Distributions.Uniform(0.1, 5), 50000)
    # Y_sim = LRMoE.sim_dataset(α, X, model, exposure = expos)

    # α_guess = fill(0.0, 2, 7)
    # model_guess = [GammaExpert(0.5*param1[1], 1.2*param2[1]) GammaExpert(0.8*param1[2], 0.8*param2[2]);
    #                GammaExpert(2*param1[3], 1.25*param2[3]) GammaExpert(0.9*param1[4], param2[4])]

    # pen_α = 5
    # pen_params = [[[1.0 Inf 1.0 Inf], [1.0 Inf 1.0 Inf]],
    #               [[1.0 Inf 1.0 Inf], [1.0 Inf 1.0 Inf]]]

    # result = LRMoE.fit_exact(Y_sim, X, α_guess, model_guess,
    #                     exposure = expos,
    #                     penalty = true, 
    #                     pen_α = pen_α, pen_params = pen_params,
    #                     # ϵ = 1e-03, α_iter_max = 5.0, ecm_iter_max = 200,
    #                     # grad_jump = true, grad_seq = nothing,
    #                     print_steps = true)
    
    # α
    # result.α_fit 
   
    # model
    # result.model_fit



end


@testset "ZIGamma, exact" begin

    # Fit 50,000 observations as an example

    # param0 = rand(Distributions.Uniform(0, 1), 4)
    # param1 = rand(Distributions.Uniform(3, 8), 4)
    # param2 = rand(Distributions.Uniform(0.5, 15), 4)

    # X = rand(Distributions.Uniform(-5, 5), 50000, 7)
    # α = rand(Distributions.Uniform(-1, 1), 2, 7)
    # α[2,:] .= 0

    # model = [ZIGammaExpert(param0[1], param1[1], param2[1]) ZIGammaExpert(param0[2], param1[2], param2[2]);
    #          ZIGammaExpert(param0[3], param1[3], param2[3]) ZIGammaExpert(param0[4], param1[4], param2[4])]

    # expos = rand(Distributions.Uniform(0.1, 5), 50000)
    # Y_sim = LRMoE.sim_dataset(α, X, model, exposure = expos)

    # α_guess = fill(0.0, 2, 7)
    # model_guess = [ZIGammaExpert(0.5, 0.5*param1[1], 1.2*param2[1]) ZIGammaExpert(0.5, 0.8*param1[2], 0.8*param2[2]);
    #                ZIGammaExpert(0.5, 2*param1[3], 1.25*param2[3]) ZIGammaExpert(0.5, 0.9*param1[4], param2[4])]

    # pen_α = 5
    # pen_params = [[[1.0 Inf 1.0 Inf], [1.0 Inf 1.0 Inf]],
    #               [[1.0 Inf 1.0 Inf], [1.0 Inf 1.0 Inf]]]

    # result = LRMoE.fit_exact(Y_sim, X, α_guess, model_guess,
    #                     exposure = expos,
    #                     penalty = true, 
    #                     pen_α = pen_α, pen_params = pen_params,
    #                     # ϵ = 1e-03, α_iter_max = 5.0, ecm_iter_max = 200,
    #                     # grad_jump = true, grad_seq = nothing,
    #                     print_steps = true)
    
    # α
    # result.α_fit 
   
    # model
    # result.model_fit



end

@testset "InverseGaussianm, exact" begin

    # Fit 50,000 observations as an example

    param0 = rand(Distributions.Uniform(0, 1), 1)[1]
    param1 = rand(Distributions.Uniform(1, 10), 4)
    param2 = rand(Distributions.Uniform(0.5, 30), 4)

    X = rand(Distributions.Uniform(-5, 5), 50000, 7)
    α = rand(Distributions.Uniform(-1, 1), 2, 7)
    α[2,:] .= 0

    model = [InverseGaussianExpert(param1[1], param2[1]) InverseGaussianExpert(param1[2], param2[2]);
             InverseGaussianExpert(param1[3], param2[3]) InverseGaussianExpert(param1[4], param2[4])]

    expos = rand(Distributions.Uniform(0.1, 5), 50000)
    Y_sim = LRMoE.sim_dataset(α, X, model, exposure = expos)

    α_guess = fill(0.0, 2, 7)
    model_guess = [InverseGaussianExpert(0.5*param1[1], 1.2*param2[1]) InverseGaussianExpert(0.75*param1[2], 0.8*param2[2]);
                   InverseGaussianExpert(2*param1[3], 1.25*param2[3]) InverseGaussianExpert(0.1*param1[4], param2[4])]

    pen_α = 5
    pen_params = [[[1.0 Inf 1.0 Inf], [1.0 Inf 1.0 Inf]],
                  [[1.0 Inf 1.0 Inf], [1.0 Inf 1.0 Inf]]]

    result = LRMoE.fit_exact(Y_sim, X, α_guess, model_guess,
                        exposure = expos,
                        penalty = true, 
                        pen_α = pen_α, pen_params = pen_params,
                        # ϵ = 1e-03, α_iter_max = 5.0, ecm_iter_max = 200,
                        # grad_jump = true, grad_seq = nothing,
                        print_steps = true)
    
    α
    result.α_fit 
   
    model
    result.model_fit



end