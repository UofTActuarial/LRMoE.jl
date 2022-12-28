using Test
using Distributions
using StatsFuns

@testset "loglik list (individual)" begin
    μμ = [1.0, 2.0]
    σσ = [1.0, 0.5]

    for μ in μμ, σ in σσ
        l = Distributions.LogNormal(μ, σ)
        y = collect(0:1:24)
        sample_size = length(y)
        expos = fill(1.0, sample_size)

        # LogNormal
        r = LRMoE.LogNormalExpert(μ, σ)

        Y = hcat(
            fill(0, sample_size),
            y,
            y,
            fill(Inf, sample_size),
            fill(0, sample_size),
            0.80 .* y,
            1.25 .* y,
            fill(Inf, sample_size),
        )

        model_raw = [
            LRMoE.LogNormalExpert(μ, σ) LRMoE.LogNormalExpert(0.5 * μ, 0.6 * σ) LRMoE.LogNormalExpert(1.5 * μ, 2.0 * σ)
            LRMoE.ZILogNormalExpert(0.4, μ, σ) LRMoE.LogNormalExpert(0.5 * μ, 0.6 * σ) LRMoE.ZILogNormalExpert(0.80, 1.5 * μ, 2.0 * σ)
        ]

        model = LRMoE.exposurize_model(model_raw; exposure=expos)

        # Test size
        @test size(expert_ll_list(Y, model)[:, :, 1]) == (2, 3)
        @test size(expert_ll_list(Y, model))[3] == sample_size

        @test size(expert_tn_list(Y, model)[:, :, 1]) == (2, 3)
        @test size(expert_tn_list(Y, model))[3] == sample_size

        @test size(expert_tn_bar_list(Y, model)[:, :, 1]) == (2, 3)
        @test size(expert_tn_bar_list(Y, model))[3] == sample_size
    end
end

@testset "loglik list (aggredated) " begin
    μμ = [1.0, 2.0]
    σσ = [1.0, 0.5]

    for μ in μμ, σ in σσ
        l = Distributions.LogNormal(μ, σ)
        y = collect(0:1:24)
        sample_size = length(y)
        expos = fill(1.0, sample_size)

        # LogNormal
        r = LRMoE.LogNormalExpert(μ, σ)

        Random.seed!(1234)

        X = rand(Uniform(-1, 1), sample_size, 10)
        α = rand(Uniform(-1, 1), 3, 10)
        α[3, :] .= 0.0

        Y = hcat(
            fill(0, sample_size),
            y,
            y,
            fill(Inf, sample_size),
            fill(0, sample_size),
            0.80 .* y,
            1.25 .* y,
            fill(Inf, sample_size),
        )
        model_raw = [
            LRMoE.LogNormalExpert(μ, σ) LRMoE.LogNormalExpert(0.5 * μ, 0.6 * σ) LRMoE.LogNormalExpert(1.5 * μ, 2.0 * σ)
            LRMoE.ZILogNormalExpert(0.4, μ, σ) LRMoE.LogNormalExpert(0.5 * μ, 0.6 * σ) LRMoE.ZILogNormalExpert(0.80, 1.5 * μ, 2.0 * σ)
        ]
        model = LRMoE.exposurize_model(model_raw; exposure=expos)

        # Test size
        @test size(LogitGating(α, X))[1] == sample_size
        @test size(LogitGating(α, X))[2] == 3

        gate = LogitGating(α, X)
        ll_ls = loglik_np(Y, gate, model)
        # Test number
        @test isa(ll_ls.ll, Number)
        @test exp.(ll_ls.gate_expert_tn) + exp.(ll_ls.gate_expert_tn_bar) ≈
            fill(1.0, length(ll_ls.gate_expert_tn))
    end
end