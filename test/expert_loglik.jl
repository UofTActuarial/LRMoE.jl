using Test
using Distributions

μ = 1
σ = 2


@testset "expert_ll_pos: LogNormal, ZILogNormal" begin

    d = Distributions.Gamma(1.0, 2.0)

    μμ = rand(d, 20)
    σσ = rand(d, 20)

    for (μ, σ) in zip(μμ, σσ)
        l = Distributions.LogNormal(μ, σ)
        x = rand(l, 100)

        # LogNormal
        r = LRMoE.LogNormalExpert(μ, σ)
        
        @test LRMoE.expert_ll_pos.(r, x, x, x, x) ≈ Distributions.logpdf.(l, x)
        @test LRMoE.expert_ll_pos.(r, 0.0, x, x, Inf) ≈ Distributions.logpdf.(l, x)
        @test LRMoE.expert_ll_pos.(r, 0.0, x, Inf, Inf) ≈ log.(Distributions.cdf.(l, Inf) .- Distributions.cdf.(l, x))
        @test LRMoE.expert_ll_pos.(r, 0.0, 0.75.*x, 1.25.*x, Inf) ≈ log.(Distributions.cdf.(l, 1.25.*x) .- Distributions.cdf.(l, 0.75.*x))
        @test LRMoE.expert_ll_pos.(r, 0.75.*x, 0.75.*x, 1.25.*x, 1.25.*x) ≈ log.(Distributions.cdf.(l, 1.25.*x) .- Distributions.cdf.(l, 0.75.*x))
        @test LRMoE.expert_ll_pos.(r, 0.50.*x, 0.75.*x, 1.25.*x, 2.00.*x) ≈ log.(Distributions.cdf.(l, 1.25.*x) .- Distributions.cdf.(l, 0.75.*x))
        @test LRMoE.expert_ll_pos.(r, 0.0, 0.0, Inf, Inf) ≈ log.(Distributions.cdf.(l, Inf) .- Distributions.cdf.(l, 0.0))

        @test LRMoE.expert_tn_pos.(r, x, x, x, x) ≈ Distributions.logpdf.(l, x)
        @test LRMoE.expert_tn_pos.(r, 0.0, x, x, Inf) ≈ fill(log.(Distributions.cdf.(l, Inf) .- Distributions.cdf.(l, 0.0)), length(x))
        @test LRMoE.expert_tn_pos.(r, 0.0, x, Inf, Inf) ≈ fill(log.(Distributions.cdf.(l, Inf) .- Distributions.cdf.(l, 0.0)), length(x))
        @test LRMoE.expert_tn_pos.(r, 0.0, 0.75.*x, 1.25.*x, Inf) ≈ fill(log.(Distributions.cdf.(l, Inf) .- Distributions.cdf.(l, 0.0)), length(x))
        @test LRMoE.expert_tn_pos.(r, 0.75.*x, 0.75.*x, 1.25.*x, 1.25.*x) ≈ log.(Distributions.cdf.(l, 1.25.*x) .- Distributions.cdf.(l, 0.75.*x))
        @test LRMoE.expert_tn_pos.(r, 0.50.*x, 0.75.*x, 1.25.*x, 2.00.*x) ≈ log.(Distributions.cdf.(l, 2.00.*x) .- Distributions.cdf.(l, 0.50.*x))
        @test LRMoE.expert_tn_pos.(r, 0.0, 0.0, Inf, Inf) ≈ log.(Distributions.cdf.(l, Inf) .- Distributions.cdf.(l, 0.0))
        @test LRMoE.expert_tn_pos.(r, 0.0, 0.0, 0.0, 0.0) ≈ -Inf

        @test LRMoE.expert_tn_bar_pos.(r, LRMoE.expert_tn_pos.(r, x, x, x, x)) ≈ log.(1.0 .- exp.(LRMoE.expert_tn_pos.(r, x, x, x, x)))
        @test LRMoE.expert_tn_bar_pos.(r, LRMoE.expert_tn_pos.(r, 0.0, x, x, Inf)) ≈ log.(1.0 .- exp.(LRMoE.expert_tn_pos.(r, 0.0, x, x, Inf)))
        @test LRMoE.expert_tn_bar_pos.(r, LRMoE.expert_tn_pos.(r, 0.0, x, Inf, Inf)) ≈ log.(1.0 .- exp.(LRMoE.expert_tn_pos.(r, 0.0, x, Inf, Inf)))
        @test LRMoE.expert_tn_bar_pos.(r, LRMoE.expert_tn_pos.(r, 0.0, 0.75.*x, 1.25.*x, Inf)) ≈ log.(1.0 .- exp.(LRMoE.expert_tn_pos.(r, 0.0, 0.75.*x, 1.25.*x, Inf)))
        @test LRMoE.expert_tn_bar_pos.(r, LRMoE.expert_tn_pos.(r, 0.75.*x, 0.75.*x, 1.25.*x, 1.25.*x)) ≈ log.(1.0 .- Distributions.cdf.(l, 1.25.*x) .+ Distributions.cdf.(l, 0.75.*x))
        @test LRMoE.expert_tn_bar_pos.(r, LRMoE.expert_tn_pos.(r, 0.50.*x, 0.75.*x, 1.25.*x, 2.00.*x)) ≈ log.(1.0 .- Distributions.cdf.(l, 2.00.*x) .+ Distributions.cdf.(l, 0.50.*x))
        @test LRMoE.expert_tn_bar_pos.(r, LRMoE.expert_tn_pos.(r, 0.0, 0.0, Inf, Inf)) ≈ log.(1.0 .- Distributions.cdf.(l, Inf) .+ Distributions.cdf.(l, 0.0))
        @test LRMoE.expert_tn_bar_pos.(r, LRMoE.expert_tn_pos.(r, 0.0, 0.0, 0.0, 0.0)) ≈ 0.0

        # ZILogNormal
        r = LRMoE.ZILogNormalExpert(0.50, μ, σ)

        @test LRMoE.expert_ll_pos.(r, x, x, x, x) ≈ Distributions.logpdf.(l, x)
        @test LRMoE.expert_ll_pos.(r, 0.0, x, x, Inf) ≈ Distributions.logpdf.(l, x)
        @test LRMoE.expert_ll_pos.(r, 0.0, x, Inf, Inf) ≈ log.(Distributions.cdf.(l, Inf) .- Distributions.cdf.(l, x))
        @test LRMoE.expert_ll_pos.(r, 0.0, 0.75.*x, 1.25.*x, Inf) ≈ log.(Distributions.cdf.(l, 1.25.*x) .- Distributions.cdf.(l, 0.75.*x))
        @test LRMoE.expert_ll_pos.(r, 0.75.*x, 0.75.*x, 1.25.*x, 1.25.*x) ≈ log.(Distributions.cdf.(l, 1.25.*x) .- Distributions.cdf.(l, 0.75.*x))
        @test LRMoE.expert_ll_pos.(r, 0.50.*x, 0.75.*x, 1.25.*x, 2.00.*x) ≈ log.(Distributions.cdf.(l, 1.25.*x) .- Distributions.cdf.(l, 0.75.*x))
        @test LRMoE.expert_ll_pos.(r, 0.0, 0.0, Inf, Inf) ≈ log.(Distributions.cdf.(l, Inf) .- Distributions.cdf.(l, 0.0))

        @test LRMoE.expert_tn_pos.(r, x, x, x, x) ≈ Distributions.logpdf.(l, x)
        @test LRMoE.expert_tn_pos.(r, 0.0, x, x, Inf) ≈ fill(log.(Distributions.cdf.(l, Inf) .- Distributions.cdf.(l, 0.0)), length(x))
        @test LRMoE.expert_tn_pos.(r, 0.0, x, Inf, Inf) ≈ fill(log.(Distributions.cdf.(l, Inf) .- Distributions.cdf.(l, 0.0)), length(x))
        @test LRMoE.expert_tn_pos.(r, 0.0, 0.75.*x, 1.25.*x, Inf) ≈ fill(log.(Distributions.cdf.(l, Inf) .- Distributions.cdf.(l, 0.0)), length(x))
        @test LRMoE.expert_tn_pos.(r, 0.75.*x, 0.75.*x, 1.25.*x, 1.25.*x) ≈ log.(Distributions.cdf.(l, 1.25.*x) .- Distributions.cdf.(l, 0.75.*x))
        @test LRMoE.expert_tn_pos.(r, 0.50.*x, 0.75.*x, 1.25.*x, 2.00.*x) ≈ log.(Distributions.cdf.(l, 2.00.*x) .- Distributions.cdf.(l, 0.50.*x))
        @test LRMoE.expert_tn_pos.(r, 0.0, 0.0, Inf, Inf) ≈ log.(Distributions.cdf.(l, Inf) .- Distributions.cdf.(l, 0.0))
        @test LRMoE.expert_tn_pos.(r, 0.0, 0.0, 0.0, 0.0) ≈ -Inf

        @test LRMoE.expert_tn_bar_pos.(r, LRMoE.expert_tn_pos.(r, x, x, x, x)) ≈ log.(1.0 .- exp.(LRMoE.expert_tn_pos.(r, x, x, x, x)))
        @test LRMoE.expert_tn_bar_pos.(r, LRMoE.expert_tn_pos.(r, 0.0, x, x, Inf)) ≈ log.(1.0 .- exp.(LRMoE.expert_tn_pos.(r, 0.0, x, x, Inf)))
        @test LRMoE.expert_tn_bar_pos.(r, LRMoE.expert_tn_pos.(r, 0.0, x, Inf, Inf)) ≈ log.(1.0 .- exp.(LRMoE.expert_tn_pos.(r, 0.0, x, Inf, Inf)))
        @test LRMoE.expert_tn_bar_pos.(r, LRMoE.expert_tn_pos.(r, 0.0, 0.75.*x, 1.25.*x, Inf)) ≈ log.(1.0 .- exp.(LRMoE.expert_tn_pos.(r, 0.0, 0.75.*x, 1.25.*x, Inf)))
        @test LRMoE.expert_tn_bar_pos.(r, LRMoE.expert_tn_pos.(r, 0.75.*x, 0.75.*x, 1.25.*x, 1.25.*x)) ≈ log.(1.0 .- Distributions.cdf.(l, 1.25.*x) .+ Distributions.cdf.(l, 0.75.*x))
        @test LRMoE.expert_tn_bar_pos.(r, LRMoE.expert_tn_pos.(r, 0.50.*x, 0.75.*x, 1.25.*x, 2.00.*x)) ≈ log.(1.0 .- Distributions.cdf.(l, 2.00.*x) .+ Distributions.cdf.(l, 0.50.*x))
        @test LRMoE.expert_tn_bar_pos.(r, LRMoE.expert_tn_pos.(r, 0.0, 0.0, Inf, Inf)) ≈ log.(1.0 .- Distributions.cdf.(l, Inf) .+ Distributions.cdf.(l, 0.0))
        @test LRMoE.expert_tn_bar_pos.(r, LRMoE.expert_tn_pos.(r, 0.0, 0.0, 0.0, 0.0)) ≈ 0.0

    end

end
