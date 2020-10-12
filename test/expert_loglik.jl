using Test
using Distributions
using StatsFuns

μ = 1
σ = 2


@testset "expert_(ll/tn/tn_bar)_pos: LogNormal, ZILogNormal" begin

    d = Distributions.Gamma(1.0, 2.0)

    μμ = rand(d, 10)
    σσ = rand(d, 10)

    for (μ, σ) in zip(μμ, σσ)
        l = Distributions.LogNormal(μ, σ)
        x = rand(l, 20)

        # LogNormal
        r = LRMoE.LogNormalExpert(μ, σ)
        
        @test isapprox(LRMoE.expert_ll_pos.(r, x, x, x, x), Distributions.logpdf.(l, x), atol = 1e-6)
        @test isapprox(LRMoE.expert_ll_pos.(r, 0.0, x, x, Inf), Distributions.logpdf.(l, x), atol = 1e-6)
        @test isapprox(LRMoE.expert_ll_pos.(r, 0.0, x, Inf, Inf), logcdf.(r, Inf) .+ log1mexp.(logcdf.(r, x) .- logcdf.(r, Inf)), atol = 1e-6)
        @test isapprox(LRMoE.expert_ll_pos.(r, 0.0, 0.75.*x, 1.25.*x, Inf), logcdf.(r, 1.25.*x) .+ log1mexp.(logcdf.(r, 0.75.*x) .- logcdf.(r, 1.25.*x)), atol = 1e-6)
        @test isapprox(LRMoE.expert_ll_pos.(r, 0.75.*x, 0.75.*x, 1.25.*x, 1.25.*x), logcdf.(r, 1.25.*x) .+ log1mexp.(logcdf.(r, 0.75.*x) .- logcdf.(r, 1.25.*x)), atol = 1e-6)
        @test isapprox(LRMoE.expert_ll_pos.(r, 0.50.*x, 0.75.*x, 1.25.*x, 2.00.*x), logcdf.(r, 1.25.*x) .+ log1mexp.(logcdf.(r, 0.75.*x) .- logcdf.(r, 1.25.*x)), atol = 1e-6)
        @test isapprox(LRMoE.expert_ll_pos.(r, 0.0, 0.0, Inf, Inf), logcdf.(r, Inf) .+ log1mexp.(logcdf.(r, 0.0) .- logcdf.(r, Inf)), atol = 1e-6)

        @test isapprox(LRMoE.expert_tn_pos.(r, x, x, x, x), Distributions.logpdf.(l, x), atol = 1e-6)
        @test isapprox(LRMoE.expert_tn_pos.(r, 0.0, x, x, Inf), fill(log.(Distributions.cdf.(l, Inf) .- Distributions.cdf.(l, 0.0)), length(x)), atol = 1e-6)
        @test isapprox(LRMoE.expert_tn_pos.(r, 0.0, x, Inf, Inf), fill(log.(Distributions.cdf.(l, Inf) .- Distributions.cdf.(l, 0.0)), length(x)), atol = 1e-6)
        @test isapprox(LRMoE.expert_tn_pos.(r, 0.0, 0.75.*x, 1.25.*x, Inf), fill(log.(Distributions.cdf.(l, Inf) .- Distributions.cdf.(l, 0.0)), length(x)), atol = 1e-6)
        @test isapprox(LRMoE.expert_tn_pos.(r, 0.75.*x, 0.75.*x, 1.25.*x, 1.25.*x), log.(Distributions.cdf.(l, 1.25.*x) .- Distributions.cdf.(l, 0.75.*x)), atol = 1e-6)
        @test isapprox(LRMoE.expert_tn_pos.(r, 0.50.*x, 0.75.*x, 1.25.*x, 2.00.*x), log.(Distributions.cdf.(l, 2.00.*x) .- Distributions.cdf.(l, 0.50.*x)), atol = 1e-6)
        @test isapprox(LRMoE.expert_tn_pos.(r, 0.0, 0.0, Inf, Inf), log.(Distributions.cdf.(l, Inf) .- Distributions.cdf.(l, 0.0)), atol = 1e-6)
        @test isapprox(LRMoE.expert_tn_pos.(r, 0.0, 0.0, 0.0, 0.0), -Inf, atol = 1e-6)

        @test isapprox(LRMoE.expert_tn_bar_pos.(r, x, x, x, x), fill(0.0, length(x)), atol = 1e-6)
        @test isapprox(LRMoE.expert_tn_bar_pos.(r, 0.0, x, x, Inf), log1mexp.(LRMoE.expert_tn_pos.(r, 0.0, x, x, Inf)), atol = 1e-6)
        @test isapprox(LRMoE.expert_tn_bar_pos.(r, 0.0, x, Inf, Inf), log1mexp.(LRMoE.expert_tn_pos.(r, 0.0, x, Inf, Inf)), atol = 1e-6)
        @test isapprox(LRMoE.expert_tn_bar_pos.(r, 0.0, 0.75.*x, 1.25.*x, Inf), log1mexp.(LRMoE.expert_tn_pos.(r, 0.0, 0.75.*x, 1.25.*x, Inf)), atol = 1e-6)
        @test isapprox(LRMoE.expert_tn_bar_pos.(r, 0.75.*x, 0.75.*x, 1.25.*x, 1.25.*x), log1mexp.(LRMoE.expert_tn_pos.(r, 0.75.*x, 0.75.*x, 1.25.*x, 1.25.*x)), atol = 1e-6)
        @test isapprox(LRMoE.expert_tn_bar_pos.(r, 0.50.*x, 0.75.*x, 1.25.*x, 2.00.*x), log1mexp.(LRMoE.expert_tn_pos.(r, 0.50.*x, 0.75.*x, 1.25.*x, 2.00.*x)), atol = 1e-6)
        @test isapprox(LRMoE.expert_tn_bar_pos.(r, 0.0, 0.0, Inf, Inf), log1mexp.(LRMoE.expert_tn_pos.(r, 0.0, 0.0, Inf, Inf)), atol = 1e-6)
        @test isapprox(LRMoE.expert_tn_bar_pos.(r, 0.0, 0.0, 0.0, 0.0), 0.0, atol = 1e-6)

        # ZILogNormal
        r = LRMoE.ZILogNormalExpert(0.50, μ, σ)

        @test isapprox(LRMoE.expert_ll_pos.(r, x, x, x, x), Distributions.logpdf.(l, x), atol = 1e-6)
        @test isapprox(LRMoE.expert_ll_pos.(r, 0.0, x, x, Inf), Distributions.logpdf.(l, x), atol = 1e-6)
        @test isapprox(LRMoE.expert_ll_pos.(r, 0.0, x, Inf, Inf), logcdf.(r, Inf) .+ log1mexp.(logcdf.(r, x) .- logcdf.(r, Inf)), atol = 1e-6)
        @test isapprox(LRMoE.expert_ll_pos.(r, 0.0, 0.75.*x, 1.25.*x, Inf), logcdf.(r, 1.25.*x) .+ log1mexp.(logcdf.(r, 0.75.*x) .- logcdf.(r, 1.25.*x)), atol = 1e-6)
        @test isapprox(LRMoE.expert_ll_pos.(r, 0.75.*x, 0.75.*x, 1.25.*x, 1.25.*x), logcdf.(r, 1.25.*x) .+ log1mexp.(logcdf.(r, 0.75.*x) .- logcdf.(r, 1.25.*x)), atol = 1e-6)
        @test isapprox(LRMoE.expert_ll_pos.(r, 0.50.*x, 0.75.*x, 1.25.*x, 2.00.*x), logcdf.(r, 1.25.*x) .+ log1mexp.(logcdf.(r, 0.75.*x) .- logcdf.(r, 1.25.*x)), atol = 1e-6)
        @test isapprox(LRMoE.expert_ll_pos.(r, 0.0, 0.0, Inf, Inf), logcdf.(r, Inf) .+ log1mexp.(logcdf.(r, 0.0) .- logcdf.(r, Inf)), atol = 1e-6)

        @test isapprox(LRMoE.expert_tn_pos.(r, x, x, x, x), Distributions.logpdf.(l, x), atol = 1e-6)
        @test isapprox(LRMoE.expert_tn_pos.(r, 0.0, x, x, Inf), fill(log.(Distributions.cdf.(l, Inf) .- Distributions.cdf.(l, 0.0)), length(x)), atol = 1e-6)
        @test isapprox(LRMoE.expert_tn_pos.(r, 0.0, x, Inf, Inf), fill(log.(Distributions.cdf.(l, Inf) .- Distributions.cdf.(l, 0.0)), length(x)), atol = 1e-6)
        @test isapprox(LRMoE.expert_tn_pos.(r, 0.0, 0.75.*x, 1.25.*x, Inf), fill(log.(Distributions.cdf.(l, Inf) .- Distributions.cdf.(l, 0.0)), length(x)), atol = 1e-6)
        @test isapprox(LRMoE.expert_tn_pos.(r, 0.75.*x, 0.75.*x, 1.25.*x, 1.25.*x), log.(Distributions.cdf.(l, 1.25.*x) .- Distributions.cdf.(l, 0.75.*x)), atol = 1e-6)
        @test isapprox(LRMoE.expert_tn_pos.(r, 0.50.*x, 0.75.*x, 1.25.*x, 2.00.*x), log.(Distributions.cdf.(l, 2.00.*x) .- Distributions.cdf.(l, 0.50.*x)), atol = 1e-6)
        @test isapprox(LRMoE.expert_tn_pos.(r, 0.0, 0.0, Inf, Inf), log.(Distributions.cdf.(l, Inf) .- Distributions.cdf.(l, 0.0)), atol = 1e-6)
        @test isapprox(LRMoE.expert_tn_pos.(r, 0.0, 0.0, 0.0, 0.0), -Inf, atol = 1e-6)

        @test isapprox(LRMoE.expert_tn_bar_pos.(r, x, x, x, x), fill(0.0, length(x)), atol = 1e-6)
        @test isapprox(LRMoE.expert_tn_bar_pos.(r, 0.0, x, x, Inf), log1mexp.(LRMoE.expert_tn_pos.(r, 0.0, x, x, Inf)), atol = 1e-6)
        @test isapprox(LRMoE.expert_tn_bar_pos.(r, 0.0, x, Inf, Inf), log1mexp.(LRMoE.expert_tn_pos.(r, 0.0, x, Inf, Inf)), atol = 1e-6)
        @test isapprox(LRMoE.expert_tn_bar_pos.(r, 0.0, 0.75.*x, 1.25.*x, Inf), log1mexp.(LRMoE.expert_tn_pos.(r, 0.0, 0.75.*x, 1.25.*x, Inf)), atol = 1e-6)
        @test isapprox(LRMoE.expert_tn_bar_pos.(r, 0.75.*x, 0.75.*x, 1.25.*x, 1.25.*x), log1mexp.(LRMoE.expert_tn_pos.(r, 0.75.*x, 0.75.*x, 1.25.*x, 1.25.*x)), atol = 1e-6)
        @test isapprox(LRMoE.expert_tn_bar_pos.(r, 0.50.*x, 0.75.*x, 1.25.*x, 2.00.*x), log1mexp.(LRMoE.expert_tn_pos.(r, 0.50.*x, 0.75.*x, 1.25.*x, 2.00.*x)), atol = 1e-6)
        @test isapprox(LRMoE.expert_tn_bar_pos.(r, 0.0, 0.0, Inf, Inf), log1mexp.(LRMoE.expert_tn_pos.(r, 0.0, 0.0, Inf, Inf)), atol = 1e-6)
        @test isapprox(LRMoE.expert_tn_bar_pos.(r, 0.0, 0.0, 0.0, 0.0), 0.0, atol = 1e-6)
    end

end
