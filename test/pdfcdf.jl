using Test
using Distributions

@testset "override pdf, cdf, etc." begin

    d = Distributions.Gamma(1.0, 2.0)

    μμ = rand(d, 50)
    σσ = rand(d, 50)

    for (μ, σ) in zip(μμ, σσ)
        l = Distributions.LogNormal(μ, σ)
        r = LRMoE.LogNormalExpert(μ, σ)
        x = rand(l, 100) 
        @test LRMoE.logpdf(r, x) ≈ Distributions.logpdf(l, x)
        @test LRMoE.logcdf(r, x) ≈ Distributions.logcdf(l, x)
        @test LRMoE.pdf(r, x) ≈ Distributions.pdf(l, x)
        @test LRMoE.cdf(r, x) ≈ Distributions.cdf(l, x)
    end

end