using Test
using Distributions
using StatsFuns

@testset "loglik list (individual)" begin

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

        # 2*1 model
        p = rand(Distributions.Uniform(0.0, 1.0), 1)[1]
        y = rand(l, 2) .+ 0.5
        Y = [0.0 0.80*y[1] 1.25*y[1] Inf 0.0 0.80*y[2] 1.25*y[2] Inf]
        model = reshape([LRMoE.LogNormalExpert(μ, σ);
                 LRMoE.ZILogNormalExpert(p, μ, σ)], (2, 1))
        @test expert_ll_pos_list(Y, model)[1] ≈ [Distributions.logcdf.(l, Y[3]) + log1mexp.(Distributions.logcdf.(l, Y[2]) - Distributions.logcdf.(l, Y[3]))]
        @test expert_ll_pos_list(Y, model)[2] ≈ [Distributions.logcdf.(l, Y[7]) + log1mexp.(Distributions.logcdf.(l, Y[6]) - Distributions.logcdf.(l, Y[7]))]

        @test expert_tn_pos_list(Y, model)[1] ≈ [Distributions.logcdf.(l, Y[4]) + log1mexp.(Distributions.logcdf.(l, Y[1]) - Distributions.logcdf.(l, Y[4]))]
        @test expert_tn_pos_list(Y, model)[2] ≈ [Distributions.logcdf.(l, Y[8]) + log1mexp.(Distributions.logcdf.(l, Y[5]) - Distributions.logcdf.(l, Y[8]))]

        @test expert_tn_bar_pos_list(Y, model)[1] ≈ log1mexp.([Distributions.logcdf.(l, Y[4]) + log1mexp.(Distributions.logcdf.(l, Y[1]) - Distributions.logcdf.(l, Y[4]))])
        @test expert_tn_bar_pos_list(Y, model)[2] ≈ log1mexp.([Distributions.logcdf.(l, Y[8]) + log1mexp.(Distributions.logcdf.(l, Y[5]) - Distributions.logcdf.(l, Y[8]))])

        @test expert_ll_list(Y, model)[1] ≈ [Distributions.logcdf.(l, Y[3]) + log1mexp.(Distributions.logcdf.(l, Y[2]) - Distributions.logcdf.(l, Y[3]))]
        @test expert_ll_list(Y, model)[2] ≈ [log(1-p) .+ Distributions.logcdf.(l, Y[7]) + log1mexp.(Distributions.logcdf.(l, Y[6]) - Distributions.logcdf.(l, Y[7]))]

        @test expert_tn_list(Y, model)[1] ≈ [Distributions.logcdf.(l, Y[4]) + log1mexp.(Distributions.logcdf.(l, Y[1]) - Distributions.logcdf.(l, Y[4]))]
        @test expert_tn_list(Y, model)[2] ≈ [Distributions.logcdf.(l, Y[8]) + log1mexp.(Distributions.logcdf.(l, Y[5]) - Distributions.logcdf.(l, Y[8]))]

        @test expert_tn_bar_list(Y, model)[1] ≈ log1mexp.([Distributions.logcdf.(l, Y[4]) + log1mexp.(Distributions.logcdf.(l, Y[1]) - Distributions.logcdf.(l, Y[4]))])
        @test expert_tn_bar_list(Y, model)[2] ≈ log1mexp.([Distributions.logcdf.(l, Y[8]) + log1mexp.(Distributions.logcdf.(l, Y[5]) - Distributions.logcdf.(l, Y[8]))])
        
        
        p = rand(Distributions.Uniform(0.0, 1.0), 1)[1]
        y = rand(l, 2) .+ 0.5
        Y = [0.0 0.80*y[1] 1.25*y[1] Inf 0.50*y[2] 0.80*y[2] 1.25*y[2] 2.50*y[2]]
        model = reshape([LRMoE.LogNormalExpert(μ, σ);
                LRMoE.ZILogNormalExpert(p, μ, σ)], (2, 1))
        @test expert_ll_pos_list(Y, model)[1] ≈ [Distributions.logcdf.(l, Y[3]) + log1mexp.(Distributions.logcdf.(l, Y[2]) - Distributions.logcdf.(l, Y[3]))]
        @test expert_ll_pos_list(Y, model)[2] ≈ [Distributions.logcdf.(l, Y[7]) + log1mexp.(Distributions.logcdf.(l, Y[6]) - Distributions.logcdf.(l, Y[7]))]

        @test expert_tn_pos_list(Y, model)[1] ≈ [Distributions.logcdf.(l, Y[4]) + log1mexp.(Distributions.logcdf.(l, Y[1]) - Distributions.logcdf.(l, Y[4]))]
        @test expert_tn_pos_list(Y, model)[2] ≈ [Distributions.logcdf.(l, Y[8]) + log1mexp.(Distributions.logcdf.(l, Y[5]) - Distributions.logcdf.(l, Y[8]))]

        @test expert_tn_bar_pos_list(Y, model)[1] ≈ log1mexp.([Distributions.logcdf.(l, Y[4]) + log1mexp.(Distributions.logcdf.(l, Y[1]) - Distributions.logcdf.(l, Y[4]))])
        @test expert_tn_bar_pos_list(Y, model)[2] ≈ log1mexp.([Distributions.logcdf.(l, Y[8]) + log1mexp.(Distributions.logcdf.(l, Y[5]) - Distributions.logcdf.(l, Y[8]))])

        @test expert_ll_list(Y, model)[1] ≈ [Distributions.logcdf.(l, Y[3]) + log1mexp.(Distributions.logcdf.(l, Y[2]) - Distributions.logcdf.(l, Y[3]))]
        @test expert_ll_list(Y, model)[2] ≈ [log(1-p) .+ Distributions.logcdf.(l, Y[7]) + log1mexp.(Distributions.logcdf.(l, Y[6]) - Distributions.logcdf.(l, Y[7]))]

        @test expert_tn_list(Y, model)[1] ≈ [Distributions.logcdf.(l, Y[4]) + log1mexp.(Distributions.logcdf.(l, Y[1]) - Distributions.logcdf.(l, Y[4]))]
        @test expert_tn_list(Y, model)[2] ≈ [log(1-p) .+ Distributions.logcdf.(l, Y[8]) + log1mexp.(Distributions.logcdf.(l, Y[5]) - Distributions.logcdf.(l, Y[8]))]

        @test expert_tn_bar_list(Y, model)[1] ≈ log1mexp.([Distributions.logcdf.(l, Y[4]) + log1mexp.(Distributions.logcdf.(l, Y[1]) - Distributions.logcdf.(l, Y[4]))])
        @test expert_tn_bar_list(Y, model)[2] ≈ [log.(p .+ (1-p) .* exp.(log1mexp.(Distributions.logcdf.(l, Y[8]) + log1mexp.(Distributions.logcdf.(l, Y[5]) - Distributions.logcdf.(l, Y[8])))))]

        # 2*3 model
        p1 = rand(Distributions.Uniform(0.0, 1.0), 1)[1]
        p2 = rand(Distributions.Uniform(0.0, 1.0), 1)[1]
        y = rand(l, 100, 2) .+ 0.5
        Y = [fill(0.0, length(y[:,1])) 0.80.*y[:,1] 1.25.*y[:,1] fill(Inf, length(y[:,1])) 0.50.*y[:,2] 0.80.*y[:,2] 1.25.*y[:,2] 2.50.*y[:,2]]
        model = [LRMoE.LogNormalExpert(μ, σ) LRMoE.ZILogNormalExpert(p1, 1.2*μ, 0.8*σ);
                LRMoE.ZILogNormalExpert(p2, 2*μ, 0.5*σ) LRMoE.LogNormalExpert(0.9*μ, 0.75*σ)]
        l1 = Distributions.LogNormal(μ, σ)
        l2 = Distributions.LogNormal(1.2*μ, 0.8*σ)
        l3 = Distributions.LogNormal(2*μ, 0.5*σ)
        l4 = Distributions.LogNormal(0.9*μ, 0.75*σ)
        
        @test expert_ll_pos_list(Y, model)[1] ≈ hcat(Distributions.logcdf.(l1, Y[:,3]) + log1mexp.(Distributions.logcdf.(l1, Y[:,2]) - Distributions.logcdf.(l1, Y[:,3])),
                                                    Distributions.logcdf.(l2, Y[:,3]) + log1mexp.(Distributions.logcdf.(l2, Y[:,2]) - Distributions.logcdf.(l2, Y[:,3])))
        @test expert_ll_pos_list(Y, model)[2] ≈ hcat(Distributions.logcdf.(l3, Y[:,7]) + log1mexp.(Distributions.logcdf.(l3, Y[:,6]) - Distributions.logcdf.(l3, Y[:,7])),
                                                    Distributions.logcdf.(l4, Y[:,7]) + log1mexp.(Distributions.logcdf.(l4, Y[:,6]) - Distributions.logcdf.(l4, Y[:,7])))

        @test expert_tn_pos_list(Y, model)[1] ≈ hcat(fill(0.0, length(Y[:,2])),
                                                    Distributions.logcdf.(l2, Y[:,4]) + log1mexp.(Distributions.logcdf.(l2, Y[:,1]) - Distributions.logcdf.(l2, Y[:,4])))
        @test expert_tn_pos_list(Y, model)[2] ≈ hcat(Distributions.logcdf.(l3, Y[:,8]) + log1mexp.(Distributions.logcdf.(l3, Y[:,5]) - Distributions.logcdf.(l3, Y[:,8])),
                                                    Distributions.logcdf.(l4, Y[:,8]) + log1mexp.(Distributions.logcdf.(l4, Y[:,5]) - Distributions.logcdf.(l4, Y[:,8])))

        @test expert_tn_bar_pos_list(Y, model)[1] ≈ hcat(fill(-Inf, length(Y[:,2])),
                                                    log1mexp.(Distributions.logcdf.(l2, Y[:,4]) + log1mexp.(Distributions.logcdf.(l2, Y[:,1]) - Distributions.logcdf.(l2, Y[:,4]))))
        @test expert_tn_bar_pos_list(Y, model)[2] ≈ hcat(log1mexp.(Distributions.logcdf.(l3, Y[:,8]) + log1mexp.(Distributions.logcdf.(l3, Y[:,5]) - Distributions.logcdf.(l3, Y[:,8]))),
                                                    log1mexp.(Distributions.logcdf.(l4, Y[:,8]) + log1mexp.(Distributions.logcdf.(l4, Y[:,5]) - Distributions.logcdf.(l4, Y[:,8]))))

        @test expert_ll_list(Y, model)[1] ≈ hcat(Distributions.logcdf.(l1, Y[:,3]) + log1mexp.(Distributions.logcdf.(l1, Y[:,2]) - Distributions.logcdf.(l1, Y[:,3])),
                                                log(1-p1) .+ Distributions.logcdf.(l2, Y[:,3]) + log1mexp.(Distributions.logcdf.(l2, Y[:,2]) - Distributions.logcdf.(l2, Y[:,3])))
        @test expert_ll_list(Y, model)[2] ≈ hcat(log(1-p2) .+ Distributions.logcdf.(l3, Y[:,7]) + log1mexp.(Distributions.logcdf.(l3, Y[:,6]) - Distributions.logcdf.(l3, Y[:,7])),
                                                    Distributions.logcdf.(l4, Y[:,7]) + log1mexp.(Distributions.logcdf.(l4, Y[:,6]) - Distributions.logcdf.(l4, Y[:,7])))

        @test expert_tn_list(Y, model)[1] ≈ hcat(fill(0.0, length(Y[:,2])),
                                                fill(0.0, length(Y[:,2])))
        # @test expert_tn_list(Y, model)[2] ≈ hcat(log(1-p2) .+ Distributions.logcdf.(l3, Y[:,8]) .+ log1mexp.(Distributions.logcdf.(l3, Y[:,5]) - Distributions.logcdf.(l3, Y[:,8])),
        #                                         Distributions.logcdf.(l4, Y[:,8]) + log1mexp.(Distributions.logcdf.(l4, Y[:,5]) - Distributions.logcdf.(l4, Y[:,8])))
        
        # @test isapprox(expert_tn_list(Y, model)[2], hcat(log(1-p2) .+ Distributions.logcdf.(l3, Y[:,8]) .+ log1mexp.(Distributions.logcdf.(l3, Y[:,5]) - Distributions.logcdf.(l3, Y[:,8])),
        #                                         Distributions.logcdf.(l4, Y[:,8]) + log1mexp.(Distributions.logcdf.(l4, Y[:,5]) - Distributions.logcdf.(l4, Y[:,8]))), atol = 1e-03)
        
        @test expert_tn_bar_list(Y, model)[1] ≈ hcat(fill(-Inf, length(Y[:,2])),
                                                    fill(-Inf, length(Y[:,2])))
        # @test expert_tn_bar_list(Y, model)[2] ≈ hcat(log.(p2.+ (1-p2) .* (1 .- exp.(Distributions.logcdf.(l3, Y[:,8]) + log1mexp.(Distributions.logcdf.(l3, Y[:,5]) - Distributions.logcdf.(l3, Y[:,8]))))),
        #                                             log1mexp.(Distributions.logcdf.(l4, Y[:,8]) + log1mexp.(Distributions.logcdf.(l4, Y[:,5]) - Distributions.logcdf.(l4, Y[:,8]))))
        
        # @test isapprox(expert_tn_bar_list(Y, model)[2], hcat(log.(p2.+ (1-p2) .* (1 .- exp.(Distributions.logcdf.(l3, Y[:,8]) + log1mexp.(Distributions.logcdf.(l3, Y[:,5]) - Distributions.logcdf.(l3, Y[:,8]))))),
        #                                             log1mexp.(Distributions.logcdf.(l4, Y[:,8]) + log1mexp.(Distributions.logcdf.(l4, Y[:,5]) - Distributions.logcdf.(l4, Y[:,8])))), atol = 1e-03)
        
    end


end

@testset "loglik list (aggredated) " begin
    d = Distributions.Gamma(1.0, 2.0)

    μμ = rand(d, 5)
    σσ = rand(d, 5)

    for (μ, σ) in zip(μμ, σσ)
        l = Distributions.LogNormal(μ, σ)
        y = rand(l, 25)

        # LogNormal
        r = LRMoE.LogNormalExpert(μ, σ)
        
        X = rand(Uniform(-1, 1), 25, 10)
        α = rand(Uniform(-1, 1), 3, 10)
        α[3,:] .= 0.0

        Y = hcat(fill(0, length(y)), y, y, fill(Inf, length(y)), fill(0, length(y)), 0.80.*y, 1.25.*y, fill(Inf, length(y)))
        model = [LRMoE.LogNormalExpert(μ, σ) LRMoE.LogNormalExpert(0.5*μ, 0.6*σ) LRMoE.LogNormalExpert(1.5*μ, 2.0*σ);
                LRMoE.ZILogNormalExpert(0.4, μ, σ) LRMoE.LogNormalExpert(0.5*μ, 0.6*σ) LRMoE.ZILogNormalExpert(0.80, 1.5*μ, 2.0*σ)]
        
        # Test size
        @test size(LogitGating(α, X))[1] == 25
        @test size(LogitGating(α, X))[2] == 3

        gate = LogitGating(α, X)
        ll_ls = loglik_np(Y, gate, model)
        # Test number
        @test isa(ll_ls.ll, Number)
        @test exp.(ll_ls.gate_expert_tn) + exp.(ll_ls.gate_expert_tn_bar) ≈ fill(1.0, length(ll_ls.gate_expert_tn))
        
    end
end