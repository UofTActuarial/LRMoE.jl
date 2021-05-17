using Test
using Distributions
using StatsFuns

using LRMoE

@testset "expert_ll: continuous, NonZI" begin

    param1 = 3.0
    param2 = 2.0
    param3 = 5.0

    l = Distributions.LogNormal(param1, param2)
    x = rand(l, 20)

    tmp = LRMoE.BurrExpert(param1, param2, param3)

    @test isapprox(LRMoE.expert_ll_exact.(tmp, x), LRMoE.logpdf.(tmp, x))

    @test isapprox(LRMoE.expert_ll.(tmp, 0.0, x, x, Inf), LRMoE.logpdf.(tmp, x))
    @test isapprox(LRMoE.expert_ll.(tmp, 0.0, 0.0, 0.0, 0.0), LRMoE.logpdf.(tmp, 0.0))
    @test isapprox(LRMoE.expert_ll.(tmp, 0.0, x, Inf, Inf), logcdf.(tmp, Inf) .+ log1mexp.(logcdf.(tmp, x) .- logcdf.(tmp, Inf)), atol = 1e-6)
    @test isapprox(LRMoE.expert_ll.(tmp, 0.0, 0, Inf, Inf), logcdf.(tmp, Inf) .+ log1mexp.(logcdf.(tmp, 0) .- logcdf.(tmp, Inf)), atol = 1e-6)
    @test isapprox(LRMoE.expert_ll.(tmp, 0.0, 0.75 .*x, 1.25 .*x, Inf), logcdf.(tmp, 1.25 .*x) .+ log1mexp.(logcdf.(tmp, 0.75 .*x) .- logcdf.(tmp, 1.25 .*x)), atol = 1e-6)

    @test isapprox(LRMoE.expert_tn.(tmp, 0.0, x, x, Inf), fill(0.0, length(x)))
    @test isapprox(LRMoE.expert_tn.(tmp, 0.0, 0.0, 0.0, 0.0), -Inf)
    @test isapprox(LRMoE.expert_tn.(tmp, 0.25 .*x, 0.75 .*x, 1.25 .*x, 1.5 .*x), logcdf.(tmp, 1.5 .*x) .+ log1mexp.(logcdf.(tmp, 0.25 .*x) .- logcdf.(tmp, 1.5 .*x)), atol = 1e-6)

    @test isapprox(LRMoE.expert_tn_bar.(tmp, 0.0, x, x, Inf), fill(-Inf, length(x)))
    @test isapprox(LRMoE.expert_tn_bar.(tmp, 0.0, 0.0, 0.0, 0.0), 0.0)
    @test isapprox(LRMoE.expert_tn_bar.(tmp, 0.25 .*x, 0.75 .*x, 1.25 .*x, 1.5 .*x), log1mexp.(logcdf.(tmp, 1.5 .*x) .+ log1mexp.(logcdf.(tmp, 0.25 .*x) .- logcdf.(tmp, 1.5 .*x))), atol = 1e-6)


    tmp = LRMoE.GammaExpert(param1, param2)

    @test isapprox(LRMoE.expert_ll_exact.(tmp, x), LRMoE.logpdf.(tmp, x))

    @test isapprox(LRMoE.expert_ll.(tmp, 0.0, x, x, Inf), LRMoE.logpdf.(tmp, x))
    @test isapprox(LRMoE.expert_ll.(tmp, 0.0, 0.0, 0.0, 0.0), LRMoE.logpdf.(tmp, 0.0))
    @test isapprox(LRMoE.expert_ll.(tmp, 0.0, x, Inf, Inf), logcdf.(tmp, Inf) .+ log1mexp.(logcdf.(tmp, x) .- logcdf.(tmp, Inf)), atol = 1e-6)
    @test isapprox(LRMoE.expert_ll.(tmp, 0.0, 0, Inf, Inf), logcdf.(tmp, Inf) .+ log1mexp.(logcdf.(tmp, 0) .- logcdf.(tmp, Inf)), atol = 1e-6)
    @test isapprox(LRMoE.expert_ll.(tmp, 0.0, 0.75 .*x, 1.25 .*x, Inf), logcdf.(tmp, 1.25 .*x) .+ log1mexp.(logcdf.(tmp, 0.75 .*x) .- logcdf.(tmp, 1.25 .*x)), atol = 1e-6)

    @test isapprox(LRMoE.expert_tn.(tmp, 0.0, x, x, Inf), fill(0.0, length(x)))
    @test isapprox(LRMoE.expert_tn.(tmp, 0.0, 0.0, 0.0, 0.0), -Inf)
    @test isapprox(LRMoE.expert_tn.(tmp, 0.25 .*x, 0.75 .*x, 1.25 .*x, 1.5 .*x), logcdf.(tmp, 1.5 .*x) .+ log1mexp.(logcdf.(tmp, 0.25 .*x) .- logcdf.(tmp, 1.5 .*x)), atol = 1e-6)

    @test isapprox(LRMoE.expert_tn_bar.(tmp, 0.0, x, x, Inf), fill(-Inf, length(x)))
    @test isapprox(LRMoE.expert_tn_bar.(tmp, 0.0, 0.0, 0.0, 0.0), 0.0)
    @test isapprox(LRMoE.expert_tn_bar.(tmp, 0.25 .*x, 0.75 .*x, 1.25 .*x, 1.5 .*x), log1mexp.(logcdf.(tmp, 1.5 .*x) .+ log1mexp.(logcdf.(tmp, 0.25 .*x) .- logcdf.(tmp, 1.5 .*x))), atol = 1e-6)


    tmp = LRMoE.InverseGaussianExpert(param1, param2)

    @test isapprox(LRMoE.expert_ll_exact.(tmp, x), LRMoE.logpdf.(tmp, x))

    @test isapprox(LRMoE.expert_ll.(tmp, 0.0, x, x, Inf), LRMoE.logpdf.(tmp, x))
    @test isapprox(LRMoE.expert_ll.(tmp, 0.0, 0.0, 0.0, 0.0), LRMoE.logpdf.(tmp, 0.0))
    @test isapprox(LRMoE.expert_ll.(tmp, 0.0, x, Inf, Inf), logcdf.(tmp, Inf) .+ log1mexp.(logcdf.(tmp, x) .- logcdf.(tmp, Inf)), atol = 1e-6)
    @test isapprox(LRMoE.expert_ll.(tmp, 0.0, 0, Inf, Inf), logcdf.(tmp, Inf) .+ log1mexp.(logcdf.(tmp, 0) .- logcdf.(tmp, Inf)), atol = 1e-6)
    @test isapprox(LRMoE.expert_ll.(tmp, 0.0, 0.75 .*x, 1.25 .*x, Inf), logcdf.(tmp, 1.25 .*x) .+ log1mexp.(logcdf.(tmp, 0.75 .*x) .- logcdf.(tmp, 1.25 .*x)), atol = 1e-6)

    @test isapprox(LRMoE.expert_tn.(tmp, 0.0, x, x, Inf), fill(0.0, length(x)))
    @test isapprox(LRMoE.expert_tn.(tmp, 0.0, 0.0, 0.0, 0.0), -Inf)
    @test isapprox(LRMoE.expert_tn.(tmp, 0.25 .*x, 0.75 .*x, 1.25 .*x, 1.5 .*x), logcdf.(tmp, 1.5 .*x) .+ log1mexp.(logcdf.(tmp, 0.25 .*x) .- logcdf.(tmp, 1.5 .*x)), atol = 1e-6)

    @test isapprox(LRMoE.expert_tn_bar.(tmp, 0.0, x, x, Inf), fill(-Inf, length(x)))
    @test isapprox(LRMoE.expert_tn_bar.(tmp, 0.0, 0.0, 0.0, 0.0), 0.0)
    @test isapprox(LRMoE.expert_tn_bar.(tmp, 0.25 .*x, 0.75 .*x, 1.25 .*x, 1.5 .*x), log1mexp.(logcdf.(tmp, 1.5 .*x) .+ log1mexp.(logcdf.(tmp, 0.25 .*x) .- logcdf.(tmp, 1.5 .*x))), atol = 1e-6)


    tmp = LRMoE.LogNormalExpert(param1, param2)

    @test isapprox(LRMoE.expert_ll_exact.(tmp, x), LRMoE.logpdf.(tmp, x))

    @test isapprox(LRMoE.expert_ll.(tmp, 0.0, x, x, Inf), LRMoE.logpdf.(tmp, x))
    @test isapprox(LRMoE.expert_ll.(tmp, 0.0, 0.0, 0.0, 0.0), LRMoE.logpdf.(tmp, 0.0))
    @test isapprox(LRMoE.expert_ll.(tmp, 0.0, x, Inf, Inf), logcdf.(tmp, Inf) .+ log1mexp.(logcdf.(tmp, x) .- logcdf.(tmp, Inf)), atol = 1e-6)
    @test isapprox(LRMoE.expert_ll.(tmp, 0.0, 0, Inf, Inf), logcdf.(tmp, Inf) .+ log1mexp.(logcdf.(tmp, 0) .- logcdf.(tmp, Inf)), atol = 1e-6)
    @test isapprox(LRMoE.expert_ll.(tmp, 0.0, 0.75 .*x, 1.25 .*x, Inf), logcdf.(tmp, 1.25 .*x) .+ log1mexp.(logcdf.(tmp, 0.75 .*x) .- logcdf.(tmp, 1.25 .*x)), atol = 1e-6)

    @test isapprox(LRMoE.expert_tn.(tmp, 0.0, x, x, Inf), fill(0.0, length(x)))
    @test isapprox(LRMoE.expert_tn.(tmp, 0.0, 0.0, 0.0, 0.0), -Inf)
    @test isapprox(LRMoE.expert_tn.(tmp, 0.25 .*x, 0.75 .*x, 1.25 .*x, 1.5 .*x), logcdf.(tmp, 1.5 .*x) .+ log1mexp.(logcdf.(tmp, 0.25 .*x) .- logcdf.(tmp, 1.5 .*x)), atol = 1e-6)

    @test isapprox(LRMoE.expert_tn_bar.(tmp, 0.0, x, x, Inf), fill(-Inf, length(x)))
    @test isapprox(LRMoE.expert_tn_bar.(tmp, 0.0, 0.0, 0.0, 0.0), 0.0)
    @test isapprox(LRMoE.expert_tn_bar.(tmp, 0.25 .*x, 0.75 .*x, 1.25 .*x, 1.5 .*x), log1mexp.(logcdf.(tmp, 1.5 .*x) .+ log1mexp.(logcdf.(tmp, 0.25 .*x) .- logcdf.(tmp, 1.5 .*x))), atol = 1e-6)


    tmp = LRMoE.WeibullExpert(param1, param2)

    @test isapprox(LRMoE.expert_ll_exact.(tmp, x), LRMoE.logpdf.(tmp, x))

    @test isapprox(LRMoE.expert_ll.(tmp, 0.0, x, x, Inf), LRMoE.logpdf.(tmp, x))
    @test isapprox(LRMoE.expert_ll.(tmp, 0.0, 0.0, 0.0, 0.0), LRMoE.logpdf.(tmp, 0.0))
    @test isapprox(LRMoE.expert_ll.(tmp, 0.0, x, Inf, Inf), logcdf.(tmp, Inf) .+ log1mexp.(logcdf.(tmp, x) .- logcdf.(tmp, Inf)), atol = 1e-6)
    @test isapprox(LRMoE.expert_ll.(tmp, 0.0, 0, Inf, Inf), logcdf.(tmp, Inf) .+ log1mexp.(logcdf.(tmp, 0) .- logcdf.(tmp, Inf)), atol = 1e-6)
    @test isapprox(LRMoE.expert_ll.(tmp, 0.0, 0.75 .*x, 1.25 .*x, Inf), logcdf.(tmp, 1.25 .*x) .+ log1mexp.(logcdf.(tmp, 0.75 .*x) .- logcdf.(tmp, 1.25 .*x)), atol = 1e-6)

    @test isapprox(LRMoE.expert_tn.(tmp, 0.0, x, x, Inf), fill(0.0, length(x)))
    @test isapprox(LRMoE.expert_tn.(tmp, 0.0, 0.0, 0.0, 0.0), -Inf)
    @test isapprox(LRMoE.expert_tn.(tmp, 0.25 .*x, 0.75 .*x, 1.25 .*x, 1.5 .*x), logcdf.(tmp, 1.5 .*x) .+ log1mexp.(logcdf.(tmp, 0.25 .*x) .- logcdf.(tmp, 1.5 .*x)), atol = 1e-6)

    @test isapprox(LRMoE.expert_tn_bar.(tmp, 0.0, x, x, Inf), fill(-Inf, length(x)))
    @test isapprox(LRMoE.expert_tn_bar.(tmp, 0.0, 0.0, 0.0, 0.0), 0.0)
    @test isapprox(LRMoE.expert_tn_bar.(tmp, 0.25 .*x, 0.75 .*x, 1.25 .*x, 1.5 .*x), log1mexp.(logcdf.(tmp, 1.5 .*x) .+ log1mexp.(logcdf.(tmp, 0.25 .*x) .- logcdf.(tmp, 1.5 .*x))), atol = 1e-6)

end
