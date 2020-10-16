using Test
using Distributions
using StatsFuns

@testset "loglik list" begin

    d = Distributions.Gamma(1.0, 2.0)

    μμ = rand(d, 5)
    σσ = rand(d, 5)

    for (μ, σ) in zip(μμ, σσ)
        l = Distributions.LogNormal(μ, σ)
        y = rand(l, 25)

        # LogNormal
        r = LRMoE.LogNormalExpert(μ, σ)
        
        # Y = hcat(y, y, y, y)
        # model = [LRMoE.LogNormalExpert(μ, σ) LRMoE.LogNormalExpert(0.5*μ, 0.6*σ) LRMoE.LogNormalExpert(1.5*μ, 2.0*σ)]

        Y = hcat(fill(0, length(y)), y, y, fill(Inf, length(y)), fill(0, length(y)), 0.80.*y, 1.25.*y, fill(Inf, length(y)))
        model = [LRMoE.LogNormalExpert(μ, σ) LRMoE.LogNormalExpert(0.5*μ, 0.6*σ) LRMoE.LogNormalExpert(1.5*μ, 2.0*σ);
                LRMoE.ZILogNormalExpert(0.4, μ, σ) LRMoE.LogNormalExpert(0.5*μ, 0.6*σ) LRMoE.ZILogNormalExpert(0.80, 1.5*μ, 2.0*σ)]
        
        # Test size
        @test size(expert_ll_pos_list(Y, model))[1] == 2
        @test size(expert_ll_pos_list(Y, model)[1])[1] == 25
        @test size(expert_ll_pos_list(Y, model)[1])[2] == 3

        @test size(expert_tn_pos_list(Y, model))[1] == 2
        @test size(expert_tn_pos_list(Y, model)[1])[1] == 25
        @test size(expert_tn_pos_list(Y, model)[1])[2] == 3

        @test size(expert_tn_bar_pos_list(Y, model))[1] == 2
        @test size(expert_tn_bar_pos_list(Y, model)[1])[1] == 25
        @test size(expert_tn_bar_pos_list(Y, model)[1])[2] == 3

        @test size(expert_ll_list(Y, model))[1] == 2
        @test size(expert_ll_list(Y, model)[1])[1] == 25
        @test size(expert_ll_list(Y, model)[1])[2] == 3

        @test size(expert_tn_list(Y, model))[1] == 2
        @test size(expert_tn_list(Y, model)[1])[1] == 25
        @test size(expert_tn_list(Y, model)[1])[2] == 3

        @test size(expert_tn_bar_list(Y, model))[1] == 2
        @test size(expert_tn_bar_list(Y, model)[1])[1] == 25
        @test size(expert_tn_bar_list(Y, model)[1])[2] == 3

        # 1*1 model
        y = rand(l, 1)
        Y = [0 y y Inf]
        model = reshape([LRMoE.LogNormalExpert(μ, σ)], (1, 1))
        @test expert_ll_pos_list(Y, model)[1] ≈ [Distributions.logpdf.(l, Y[2])]
        @test expert_tn_pos_list(Y, model)[1] ≈ [0.0]
        @test expert_tn_bar_pos_list(Y, model)[1] ≈ [-Inf]
        @test expert_ll_list(Y, model)[1] ≈ [Distributions.logpdf.(l, Y[2])]
        @test expert_tn_list(Y, model)[1] ≈ [0.0]
        @test expert_tn_bar_list(Y, model)[1] ≈ [-Inf]

        y = rand(l, 1)
        Y = [0 0.80*y 1.25*y Inf]
        model = reshape([LRMoE.LogNormalExpert(μ, σ)], (1, 1))
        @test expert_ll_pos_list(Y, model)[1] ≈ [Distributions.logcdf.(l, Y[3]) + log1mexp(Distributions.logcdf.(l, Y[2]) - Distributions.logcdf.(l, Y[3]))]
        @test expert_tn_pos_list(Y, model)[1] ≈ [0.0]
        @test expert_tn_bar_pos_list(Y, model)[1] ≈ [-Inf]
        @test expert_ll_list(Y, model)[1] ≈ [Distributions.logcdf.(l, Y[3]) + log1mexp(Distributions.logcdf.(l, Y[2]) - Distributions.logcdf.(l, Y[3]))]
        @test expert_tn_list(Y, model)[1] ≈ [0.0]
        @test expert_tn_bar_list(Y, model)[1] ≈ [-Inf]

        y = rand(l, 1)
        Y = [0.50*y 0.80*y 1.25*y 2.00*y]
        model = reshape([LRMoE.LogNormalExpert(μ, σ)], (1, 1))
        @test expert_ll_pos_list(Y, model)[1] ≈ [Distributions.logcdf.(l, Y[3]) + log1mexp.(Distributions.logcdf.(l, Y[2]) - Distributions.logcdf.(l, Y[3]))]
        @test expert_tn_pos_list(Y, model)[1] ≈ [Distributions.logcdf.(l, Y[4]) + log1mexp.(Distributions.logcdf.(l, Y[1]) - Distributions.logcdf.(l, Y[4]))]
        @test expert_tn_bar_pos_list(Y, model)[1] ≈ log1mexp.([Distributions.logcdf.(l, Y[4]) + log1mexp(Distributions.logcdf.(l, Y[1]) - Distributions.logcdf.(l, Y[4]))])
        @test expert_ll_list(Y, model)[1] ≈ [Distributions.logcdf.(l, Y[3]) + log1mexp.(Distributions.logcdf.(l, Y[2]) - Distributions.logcdf.(l, Y[3]))]
        @test expert_tn_list(Y, model)[1] ≈ [Distributions.logcdf.(l, Y[4]) + log1mexp.(Distributions.logcdf.(l, Y[1]) - Distributions.logcdf.(l, Y[4]))]
        @test expert_tn_bar_list(Y, model)[1] ≈ log1mexp.([Distributions.logcdf.(l, Y[4]) + log1mexp.(Distributions.logcdf.(l, Y[1]) - Distributions.logcdf.(l, Y[4]))])

        # 1*2 model
        p = rand(Distributions.Uniform(0.0, 1.0), 1)[1]
        y = rand(l, 2) .+ 0.5
        Y = [0.0 0.80*y[1] 1.25*y[1] Inf 0.0 0.80*y[2] 1.25*y[2] Inf]
        model = [LRMoE.LogNormalExpert(μ, σ) LRMoE.ZILogNormalExpert(p, μ, σ)]
        @test expert_ll_pos_list(Y, model)[1] ≈ reshape([Distributions.logcdf.(l, Y[3]) + log1mexp.(Distributions.logcdf.(l, Y[2]) - Distributions.logcdf.(l, Y[3]))
                                                        Distributions.logcdf.(l, Y[7]) + log1mexp.(Distributions.logcdf.(l, Y[6]) - Distributions.logcdf.(l, Y[7]))], (1, 2))
        @test expert_tn_pos_list(Y, model)[1] ≈ reshape([Distributions.logcdf.(l, Y[4]) + log1mexp.(Distributions.logcdf.(l, Y[1]) - Distributions.logcdf.(l, Y[4]))
                                                        Distributions.logcdf.(l, Y[8]) + log1mexp.(Distributions.logcdf.(l, Y[5]) - Distributions.logcdf.(l, Y[8]))], (1, 2))
        @test expert_tn_bar_pos_list(Y, model)[1] ≈ log1mexp.(reshape([Distributions.logcdf.(l, Y[4]) + log1mexp.(Distributions.logcdf.(l, Y[1]) - Distributions.logcdf.(l, Y[4]))
                                                        Distributions.logcdf.(l, Y[8]) + log1mexp.(Distributions.logcdf.(l, Y[5]) - Distributions.logcdf.(l, Y[8]))], (1, 2)))
        @test expert_ll_list(Y, model)[1] ≈ reshape([Distributions.logcdf.(l, Y[3]) + log1mexp.(Distributions.logcdf.(l, Y[2]) - Distributions.logcdf.(l, Y[3]))
                                                log(1-p) .+ Distributions.logcdf.(l, Y[7]) + log1mexp.(Distributions.logcdf.(l, Y[6]) - Distributions.logcdf.(l, Y[7]))], (1, 2))
        @test expert_tn_list(Y, model)[1] ≈ reshape([Distributions.logcdf.(l, Y[4]) + log1mexp.(Distributions.logcdf.(l, Y[1]) - Distributions.logcdf.(l, Y[4]))
                                                    Distributions.logcdf.(l, Y[8]) + log1mexp.(Distributions.logcdf.(l, Y[5]) - Distributions.logcdf.(l, Y[8]))], (1, 2))
        @test expert_tn_bar_list(Y, model)[1] ≈ log1mexp.(reshape([Distributions.logcdf.(l, Y[4]) + log1mexp.(Distributions.logcdf.(l, Y[1]) - Distributions.logcdf.(l, Y[4]))
                                                Distributions.logcdf.(l, Y[8]) + log1mexp.(Distributions.logcdf.(l, Y[5]) - Distributions.logcdf.(l, Y[8]))], (1, 2)))
        
        p = rand(Distributions.Uniform(0.0, 1.0), 1)[1]
        y = rand(l, 2) .+ 0.5
        Y = [0.0 0.80*y[1] 1.25*y[1] Inf 0.50*y[2] 0.80*y[2] 1.25*y[2] 2.50*y[2]]
        model = [LRMoE.LogNormalExpert(μ, σ) LRMoE.ZILogNormalExpert(p, μ, σ)]
        @test expert_ll_pos_list(Y, model)[1] ≈ reshape([Distributions.logcdf.(l, Y[3]) + log1mexp.(Distributions.logcdf.(l, Y[2]) - Distributions.logcdf.(l, Y[3]))
                                                        Distributions.logcdf.(l, Y[7]) + log1mexp.(Distributions.logcdf.(l, Y[6]) - Distributions.logcdf.(l, Y[7]))], (1, 2))
        @test expert_tn_pos_list(Y, model)[1] ≈ reshape([Distributions.logcdf.(l, Y[4]) + log1mexp.(Distributions.logcdf.(l, Y[1]) - Distributions.logcdf.(l, Y[4]))
                                                        Distributions.logcdf.(l, Y[8]) + log1mexp.(Distributions.logcdf.(l, Y[5]) - Distributions.logcdf.(l, Y[8]))], (1, 2))
        @test expert_tn_bar_pos_list(Y, model)[1] ≈ log1mexp.(reshape([Distributions.logcdf.(l, Y[4]) + log1mexp.(Distributions.logcdf.(l, Y[1]) - Distributions.logcdf.(l, Y[4]))
                                                        Distributions.logcdf.(l, Y[8]) + log1mexp.(Distributions.logcdf.(l, Y[5]) - Distributions.logcdf.(l, Y[8]))], (1, 2)))
        @test expert_ll_list(Y, model)[1] ≈ reshape([Distributions.logcdf.(l, Y[3]) + log1mexp.(Distributions.logcdf.(l, Y[2]) - Distributions.logcdf.(l, Y[3]))
                                                log(1-p) .+ Distributions.logcdf.(l, Y[7]) + log1mexp.(Distributions.logcdf.(l, Y[6]) - Distributions.logcdf.(l, Y[7]))], (1, 2))
        @test expert_tn_list(Y, model)[1] ≈ reshape([Distributions.logcdf.(l, Y[4]) + log1mexp.(Distributions.logcdf.(l, Y[1]) - Distributions.logcdf.(l, Y[4]))
                                                log(1-p) .+ Distributions.logcdf.(l, Y[8]) + log1mexp.(Distributions.logcdf.(l, Y[5]) - Distributions.logcdf.(l, Y[8]))], (1, 2))
        @test expert_tn_bar_list(Y, model)[1] ≈ log1mexp.(reshape([Distributions.logcdf.(l, Y[4]) + log1mexp.(Distributions.logcdf.(l, Y[1]) - Distributions.logcdf.(l, Y[4]))
                                                log(1-p) .+ Distributions.logcdf.(l, Y[8]) + log1mexp.(Distributions.logcdf.(l, Y[5]) - Distributions.logcdf.(l, Y[8]))], (1, 2)))


    end


end