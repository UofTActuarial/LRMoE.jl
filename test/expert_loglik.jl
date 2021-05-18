using Test
using Distributions
using StatsFuns

using LRMoE

@testset "count with exposures" begin

    mean(LRMoE.GammaCountExpert(1.0, 1.0))
    mean(LRMoE.GammaCountExpert(2.0, 1.0))
    mean(LRMoE.GammaCountExpert(3.0, 1.0))
    mean(LRMoE.GammaCountExpert(4.0, 1.0))
    mean(LRMoE.GammaCountExpert(5.0, 1.0))

    mean(LRMoE.GammaCountExpert(1.0, 2.0))
    mean(LRMoE.GammaCountExpert(2.0, 2.0))
    mean(LRMoE.GammaCountExpert(3.0, 2.0))
    mean(LRMoE.GammaCountExpert(4.0, 2.0))
    mean(LRMoE.GammaCountExpert(5.0, 2.0))

    mean(LRMoE.GammaCountExpert(1.0, 3.0))
    mean(LRMoE.GammaCountExpert(2.0, 3.0))
    mean(LRMoE.GammaCountExpert(3.0, 3.0))
    mean(LRMoE.GammaCountExpert(4.0, 3.0))
    mean(LRMoE.GammaCountExpert(5.0, 3.0))

    mean(LRMoE.GammaCountExpert(1.0, 1/1.0))
    mean(LRMoE.GammaCountExpert(1.0, 1/2.0))
    mean(LRMoE.GammaCountExpert(1.0, 1/3.0))
    mean(LRMoE.GammaCountExpert(1.0, 1/4.0))
    mean(LRMoE.GammaCountExpert(1.0, 1/5.0))
    mean(LRMoE.GammaCountExpert(1.0, 1/6.0))
    mean(LRMoE.GammaCountExpert(1.0, 1/7.0))
    mean(LRMoE.GammaCountExpert(1.0, 1/8.0))
    mean(LRMoE.GammaCountExpert(1.0, 1/9.0))

    var(LRMoE.GammaCountExpert(1.0, 1/1.0))
    var(LRMoE.GammaCountExpert(1.0, 1/2.0))
    var(LRMoE.GammaCountExpert(1.0, 1/3.0))
    var(LRMoE.GammaCountExpert(1.0, 1/4.0))
    var(LRMoE.GammaCountExpert(1.0, 1/5.0))
    var(LRMoE.GammaCountExpert(1.0, 1/6.0))
    var(LRMoE.GammaCountExpert(1.0, 1/7.0))
    var(LRMoE.GammaCountExpert(1.0, 1/8.0))
    var(LRMoE.GammaCountExpert(1.0, 1/9.0))

    mean(LRMoE.GammaCountExpert(2.0, 1/1.0))
    mean(LRMoE.GammaCountExpert(2.0, 1/2.0))
    mean(LRMoE.GammaCountExpert(2.0, 1/3.0))
    mean(LRMoE.GammaCountExpert(2.0, 1/4.0))
    mean(LRMoE.GammaCountExpert(2.0, 1/5.0))
    mean(LRMoE.GammaCountExpert(2.0, 1/6.0))
    mean(LRMoE.GammaCountExpert(2.0, 1/7.0))
    mean(LRMoE.GammaCountExpert(2.0, 1/8.0))
    mean(LRMoE.GammaCountExpert(2.0, 1/9.0))

    var(LRMoE.GammaCountExpert(2.0, 1/1.0))
    var(LRMoE.GammaCountExpert(2.0, 1/2.0))
    var(LRMoE.GammaCountExpert(2.0, 1/3.0))
    var(LRMoE.GammaCountExpert(2.0, 1/4.0))
    var(LRMoE.GammaCountExpert(2.0, 1/5.0))
    var(LRMoE.GammaCountExpert(2.0, 1/6.0))
    var(LRMoE.GammaCountExpert(2.0, 1/7.0))
    var(LRMoE.GammaCountExpert(2.0, 1/8.0))
    var(LRMoE.GammaCountExpert(2.0, 1/9.0))

    mean(LRMoE.NegativeBinomialExpert(1.0, 0.20))
    mean(LRMoE.NegativeBinomialExpert(2.0, 0.20))
    mean(LRMoE.NegativeBinomialExpert(3.0, 0.20))
    mean(LRMoE.NegativeBinomialExpert(4.0, 0.20))
    mean(LRMoE.NegativeBinomialExpert(5.0, 0.20))
    mean(LRMoE.NegativeBinomialExpert(6.0, 0.20))
    mean(LRMoE.NegativeBinomialExpert(7.0, 0.20))
    mean(LRMoE.NegativeBinomialExpert(8.0, 0.20))
    mean(LRMoE.NegativeBinomialExpert(9.0, 0.20))

    var(LRMoE.NegativeBinomialExpert(1.0, 0.20))
    var(LRMoE.NegativeBinomialExpert(2.0, 0.20))
    var(LRMoE.NegativeBinomialExpert(3.0, 0.20))
    var(LRMoE.NegativeBinomialExpert(4.0, 0.20))
    var(LRMoE.NegativeBinomialExpert(5.0, 0.20))
    var(LRMoE.NegativeBinomialExpert(6.0, 0.20))
    var(LRMoE.NegativeBinomialExpert(7.0, 0.20))
    var(LRMoE.NegativeBinomialExpert(8.0, 0.20))
    var(LRMoE.NegativeBinomialExpert(9.0, 0.20))

end




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



@testset "expert_ll: continuous, ZI" begin

    param0 = rand(Distributions.Uniform(0, 1), 1)[1]
    param1 = 3.0
    param2 = 2.0
    param3 = 5.0

    l = Distributions.LogNormal(param1, param2)
    x = rand(l, 20) .+ 0.05

    tmp = LRMoE.ZIBurrExpert(param0, param1, param2, param3)

    @test isapprox(LRMoE.expert_ll_exact.(tmp, x), log(1-param0) .+ LRMoE.logpdf.(tmp, x))
    @test isapprox(LRMoE.expert_ll_exact.(tmp, 0), log(param0))

    @test isapprox(LRMoE.expert_ll.(tmp, 0.0, x, x, Inf), log.((1-param0).*exp.(LRMoE.logpdf.(tmp, x))))
    @test isapprox(LRMoE.expert_ll.(tmp, 0.0, 0.0, 0.0, 0.0), log.(param0))
    @test isapprox(LRMoE.expert_ll.(tmp, 0.0, x, Inf, Inf), log.((1-param0).*exp.(logcdf.(tmp, Inf) .+ log1mexp.(logcdf.(tmp, x) .- logcdf.(tmp, Inf)))), atol = 1e-6)
    @test isapprox(LRMoE.expert_ll.(tmp, 0.0, 0, Inf, Inf), log.(1), atol = 1e-6)
    @test isapprox(LRMoE.expert_ll.(tmp, 0.0, 0.75 .*x, 1.25 .*x, Inf), log.((1-param0).*exp.(logcdf.(tmp, 1.25 .*x) .+ log1mexp.(logcdf.(tmp, 0.75 .*x) .- logcdf.(tmp, 1.25 .*x)))), atol = 1e-6)

    @test isapprox(LRMoE.expert_tn.(tmp, 0.0, x, x, Inf), fill(0.0, length(x)))
    @test isapprox(LRMoE.expert_tn.(tmp, 0.0, 0.0, 0.0, 0.0), log(param0))
    @test isapprox(LRMoE.expert_tn.(tmp, 0.25 .*x, 0.75 .*x, 1.25 .*x, 1.5 .*x), log.((1-param0).*exp.(logcdf.(tmp, 1.5 .*x) .+ log1mexp.(logcdf.(tmp, 0.25 .*x) .- logcdf.(tmp, 1.5 .*x)))), atol = 1e-6)

    @test isapprox(LRMoE.expert_tn_bar.(tmp, 0.0, x, x, Inf), fill(-Inf, length(x)))
    @test isapprox(LRMoE.expert_tn_bar.(tmp, 0.0, 0.0, 0.0, 0.0), log(1-param0))
    @test isapprox(LRMoE.expert_tn_bar.(tmp, 0.25 .*x, 0.75 .*x, 1.25 .*x, 1.5 .*x), log.(param0 .+ (1-param0).*exp.(log1mexp.(logcdf.(tmp, 1.5 .*x) .+ log1mexp.(logcdf.(tmp, 0.25 .*x) .- logcdf.(tmp, 1.5 .*x))))), atol = 1e-6)


    tmp = LRMoE.ZIGammaExpert(param0, param1, param2)

    @test isapprox(LRMoE.expert_ll_exact.(tmp, x), log(1-param0) .+ LRMoE.logpdf.(tmp, x))
    @test isapprox(LRMoE.expert_ll_exact.(tmp, 0), log(param0))

    @test isapprox(LRMoE.expert_ll.(tmp, 0.0, x, x, Inf), log.((1-param0).*exp.(LRMoE.logpdf.(tmp, x))))
    @test isapprox(LRMoE.expert_ll.(tmp, 0.0, 0.0, 0.0, 0.0), log.(param0))
    @test isapprox(LRMoE.expert_ll.(tmp, 0.0, x, Inf, Inf), log.((1-param0).*exp.(logcdf.(tmp, Inf) .+ log1mexp.(logcdf.(tmp, x) .- logcdf.(tmp, Inf)))), atol = 1e-6)
    @test isapprox(LRMoE.expert_ll.(tmp, 0.0, 0, Inf, Inf), log.(1), atol = 1e-6)
    @test isapprox(LRMoE.expert_ll.(tmp, 0.0, 0.75 .*x, 1.25 .*x, Inf), log.((1-param0).*exp.(logcdf.(tmp, 1.25 .*x) .+ log1mexp.(logcdf.(tmp, 0.75 .*x) .- logcdf.(tmp, 1.25 .*x)))), atol = 1e-6)

    @test isapprox(LRMoE.expert_tn.(tmp, 0.0, x, x, Inf), fill(0.0, length(x)))
    @test isapprox(LRMoE.expert_tn.(tmp, 0.0, 0.0, 0.0, 0.0), log(param0))
    @test isapprox(LRMoE.expert_tn.(tmp, 0.25 .*x, 0.75 .*x, 1.25 .*x, 1.5 .*x), log.((1-param0).*exp.(logcdf.(tmp, 1.5 .*x) .+ log1mexp.(logcdf.(tmp, 0.25 .*x) .- logcdf.(tmp, 1.5 .*x)))), atol = 1e-6)

    @test isapprox(LRMoE.expert_tn_bar.(tmp, 0.0, x, x, Inf), fill(-Inf, length(x)))
    @test isapprox(LRMoE.expert_tn_bar.(tmp, 0.0, 0.0, 0.0, 0.0), log(1-param0))
    @test isapprox(LRMoE.expert_tn_bar.(tmp, 0.25 .*x, 0.75 .*x, 1.25 .*x, 1.5 .*x), log.(param0 .+ (1-param0).*exp.(log1mexp.(logcdf.(tmp, 1.5 .*x) .+ log1mexp.(logcdf.(tmp, 0.25 .*x) .- logcdf.(tmp, 1.5 .*x))))), atol = 1e-6)


    tmp = LRMoE.ZIInverseGaussianExpert(param0, param1, param2)

    @test isapprox(LRMoE.expert_ll_exact.(tmp, x), log(1-param0) .+ LRMoE.logpdf.(tmp, x))
    @test isapprox(LRMoE.expert_ll_exact.(tmp, 0), log(param0))

    @test isapprox(LRMoE.expert_ll.(tmp, 0.0, x, x, Inf), log.((1-param0).*exp.(LRMoE.logpdf.(tmp, x))))
    @test isapprox(LRMoE.expert_ll.(tmp, 0.0, 0.0, 0.0, 0.0), log.(param0))
    @test isapprox(LRMoE.expert_ll.(tmp, 0.0, x, Inf, Inf), log.((1-param0).*exp.(logcdf.(tmp, Inf) .+ log1mexp.(logcdf.(tmp, x) .- logcdf.(tmp, Inf)))), atol = 1e-6)
    @test isapprox(LRMoE.expert_ll.(tmp, 0.0, 0, Inf, Inf), log.(1), atol = 1e-6)
    @test isapprox(LRMoE.expert_ll.(tmp, 0.0, 0.75 .*x, 1.25 .*x, Inf), log.((1-param0).*exp.(logcdf.(tmp, 1.25 .*x) .+ log1mexp.(logcdf.(tmp, 0.75 .*x) .- logcdf.(tmp, 1.25 .*x)))), atol = 1e-6)

    @test isapprox(LRMoE.expert_tn.(tmp, 0.0, x, x, Inf), fill(0.0, length(x)))
    @test isapprox(LRMoE.expert_tn.(tmp, 0.0, 0.0, 0.0, 0.0), log(param0))
    @test isapprox(LRMoE.expert_tn.(tmp, 0.25 .*x, 0.75 .*x, 1.25 .*x, 1.5 .*x), log.((1-param0).*exp.(logcdf.(tmp, 1.5 .*x) .+ log1mexp.(logcdf.(tmp, 0.25 .*x) .- logcdf.(tmp, 1.5 .*x)))), atol = 1e-6)

    @test isapprox(LRMoE.expert_tn_bar.(tmp, 0.0, x, x, Inf), fill(-Inf, length(x)))
    @test isapprox(LRMoE.expert_tn_bar.(tmp, 0.0, 0.0, 0.0, 0.0), log(1-param0))
    @test isapprox(LRMoE.expert_tn_bar.(tmp, 0.25 .*x, 0.75 .*x, 1.25 .*x, 1.5 .*x), log.(param0 .+ (1-param0).*exp.(log1mexp.(logcdf.(tmp, 1.5 .*x) .+ log1mexp.(logcdf.(tmp, 0.25 .*x) .- logcdf.(tmp, 1.5 .*x))))), atol = 1e-6)


    tmp = LRMoE.ZILogNormalExpert(param0, param1, param2)

    @test isapprox(LRMoE.expert_ll_exact.(tmp, x), log(1-param0) .+ LRMoE.logpdf.(tmp, x))
    @test isapprox(LRMoE.expert_ll_exact.(tmp, 0), log(param0))

    @test isapprox(LRMoE.expert_ll.(tmp, 0.0, x, x, Inf), log.((1-param0).*exp.(LRMoE.logpdf.(tmp, x))))
    @test isapprox(LRMoE.expert_ll.(tmp, 0.0, 0.0, 0.0, 0.0), log.(param0))
    @test isapprox(LRMoE.expert_ll.(tmp, 0.0, x, Inf, Inf), log.((1-param0).*exp.(logcdf.(tmp, Inf) .+ log1mexp.(logcdf.(tmp, x) .- logcdf.(tmp, Inf)))), atol = 1e-6)
    @test isapprox(LRMoE.expert_ll.(tmp, 0.0, 0, Inf, Inf), log.(1), atol = 1e-6)
    @test isapprox(LRMoE.expert_ll.(tmp, 0.0, 0.75 .*x, 1.25 .*x, Inf), log.((1-param0).*exp.(logcdf.(tmp, 1.25 .*x) .+ log1mexp.(logcdf.(tmp, 0.75 .*x) .- logcdf.(tmp, 1.25 .*x)))), atol = 1e-6)

    @test isapprox(LRMoE.expert_tn.(tmp, 0.0, x, x, Inf), fill(0.0, length(x)))
    @test isapprox(LRMoE.expert_tn.(tmp, 0.0, 0.0, 0.0, 0.0), log(param0))
    @test isapprox(LRMoE.expert_tn.(tmp, 0.25 .*x, 0.75 .*x, 1.25 .*x, 1.5 .*x), log.((1-param0).*exp.(logcdf.(tmp, 1.5 .*x) .+ log1mexp.(logcdf.(tmp, 0.25 .*x) .- logcdf.(tmp, 1.5 .*x)))), atol = 1e-6)

    @test isapprox(LRMoE.expert_tn_bar.(tmp, 0.0, x, x, Inf), fill(-Inf, length(x)))
    @test isapprox(LRMoE.expert_tn_bar.(tmp, 0.0, 0.0, 0.0, 0.0), log(1-param0))
    @test isapprox(LRMoE.expert_tn_bar.(tmp, 0.25 .*x, 0.75 .*x, 1.25 .*x, 1.5 .*x), log.(param0 .+ (1-param0).*exp.(log1mexp.(logcdf.(tmp, 1.5 .*x) .+ log1mexp.(logcdf.(tmp, 0.25 .*x) .- logcdf.(tmp, 1.5 .*x))))), atol = 1e-6)


    tmp = LRMoE.ZIWeibullExpert(param0, param1, param2)

    @test isapprox(LRMoE.expert_ll_exact.(tmp, x), log(1-param0) .+ LRMoE.logpdf.(tmp, x))
    @test isapprox(LRMoE.expert_ll_exact.(tmp, 0), log(param0))

    @test isapprox(LRMoE.expert_ll.(tmp, 0.0, x, x, Inf), log.((1-param0).*exp.(LRMoE.logpdf.(tmp, x))))
    @test isapprox(LRMoE.expert_ll.(tmp, 0.0, 0.0, 0.0, 0.0), log.(param0))
    @test isapprox(LRMoE.expert_ll.(tmp, 0.0, x, Inf, Inf), log.((1-param0).*exp.(logcdf.(tmp, Inf) .+ log1mexp.(logcdf.(tmp, x) .- logcdf.(tmp, Inf)))), atol = 1e-6)
    @test isapprox(LRMoE.expert_ll.(tmp, 0.0, 0, Inf, Inf), log.(1), atol = 1e-6)
    @test isapprox(LRMoE.expert_ll.(tmp, 0.0, 0.75 .*x, 1.25 .*x, Inf), log.((1-param0).*exp.(logcdf.(tmp, 1.25 .*x) .+ log1mexp.(logcdf.(tmp, 0.75 .*x) .- logcdf.(tmp, 1.25 .*x)))), atol = 1e-6)

    @test isapprox(LRMoE.expert_tn.(tmp, 0.0, x, x, Inf), fill(0.0, length(x)))
    @test isapprox(LRMoE.expert_tn.(tmp, 0.0, 0.0, 0.0, 0.0), log(param0))
    @test isapprox(LRMoE.expert_tn.(tmp, 0.25 .*x, 0.75 .*x, 1.25 .*x, 1.5 .*x), log.((1-param0).*exp.(logcdf.(tmp, 1.5 .*x) .+ log1mexp.(logcdf.(tmp, 0.25 .*x) .- logcdf.(tmp, 1.5 .*x)))), atol = 1e-6)

    @test isapprox(LRMoE.expert_tn_bar.(tmp, 0.0, x, x, Inf), fill(-Inf, length(x)))
    @test isapprox(LRMoE.expert_tn_bar.(tmp, 0.0, 0.0, 0.0, 0.0), log(1-param0))
    @test isapprox(LRMoE.expert_tn_bar.(tmp, 0.25 .*x, 0.75 .*x, 1.25 .*x, 1.5 .*x), log.(param0 .+ (1-param0).*exp.(log1mexp.(logcdf.(tmp, 1.5 .*x) .+ log1mexp.(logcdf.(tmp, 0.25 .*x) .- logcdf.(tmp, 1.5 .*x))))), atol = 1e-6)

end



@testset "expert_ll: discrete, NonZI" begin

    param1 = 20
    param2 = rand(Distributions.Uniform(0, 1), 1)[1]
    param3 = 5.0

    l = Distributions.Binomial(param1, param2)
    x = rand(l, 20)

    tmp = LRMoE.BinomialExpert(param1, param2)

    @test isapprox(LRMoE.expert_ll_exact.(tmp, x), LRMoE.logpdf.(tmp, x))

    @test isapprox(LRMoE.expert_ll.(tmp, 0.0, x, x, Inf), LRMoE.logpdf.(tmp, x))
    @test isapprox(LRMoE.expert_ll.(tmp, 0.0, 0.0, 0.0, 0.0), LRMoE.logpdf.(tmp, 0.0))
    @test isapprox(LRMoE.expert_ll.(tmp, 0.0, x, Inf, Inf), logcdf.(tmp, Inf) .+ log1mexp.(logcdf.(tmp, ceil.(x) .- 1) .- logcdf.(tmp, Inf)), atol = 1e-6)
    @test isapprox(LRMoE.expert_ll.(tmp, 0.0, 0, Inf, Inf), logcdf.(tmp, Inf) .+ log1mexp.(logcdf.(tmp, -1) .- logcdf.(tmp, Inf)), atol = 1e-6)
    @test isapprox(LRMoE.expert_ll.(tmp, 0.0, 0.75 .*x, 1.25 .*x, Inf), logcdf.(tmp, 1.25 .*x) .+ log1mexp.(logcdf.(tmp, ceil.(0.75 .*x) .- 1) .- logcdf.(tmp, 1.25 .*x)), atol = 1e-6)

    @test isapprox(LRMoE.expert_tn.(tmp, 0.0, x, x, Inf), fill(0.0, length(x)), atol = 1e-06)
    @test isapprox(LRMoE.expert_tn.(tmp, 0.0, 0.0, 0.0, 0.0), LRMoE.logpdf.(tmp, 0))
    @test isapprox(LRMoE.expert_tn.(tmp, 0.25 .*x, 0.75 .*x, 1.25 .*x, 1.5 .*x), logcdf.(tmp, 1.5 .*x) .+ log1mexp.(logcdf.(tmp, ceil.(0.25 .*x) .- 1) .- logcdf.(tmp, 1.5 .*x)), atol = 1e-6)

    # @test isapprox(LRMoE.expert_tn_bar.(tmp, 0.0, x, x, Inf), fill(-Inf, length(x)))
    @test isapprox(LRMoE.expert_tn_bar.(tmp, 0.0, 0.0, 0.0, 0.0), log1mexp(LRMoE.logpdf.(tmp, 0)), atol = 1e-06)
    @test isapprox(LRMoE.expert_tn_bar.(tmp, 0.25 .*x, 0.75 .*x, 1.25 .*x, 1.5 .*x), log1mexp.(logcdf.(tmp, 1.5 .*x) .+ log1mexp.(logcdf.(tmp, ceil.(0.25 .*x) .- 1) .- logcdf.(tmp, 1.5 .*x))), atol = 1e-6)

    tmp = LRMoE.GammaCountExpert(param1, param3)

    @test isapprox(LRMoE.expert_ll_exact.(tmp, x), LRMoE.logpdf.(tmp, x))

    @test isapprox(LRMoE.expert_ll.(tmp, 0.0, x, x, Inf), LRMoE.logpdf.(tmp, x))
    @test isapprox(LRMoE.expert_ll.(tmp, 0.0, 0.0, 0.0, 0.0), LRMoE.logpdf.(tmp, 0.0))
    @test isapprox(LRMoE.expert_ll.(tmp, 0.0, x, Inf, Inf), logcdf.(tmp, Inf) .+ log1mexp.(logcdf.(tmp, ceil.(x) .- 1) .- logcdf.(tmp, Inf)), atol = 1e-6)
    @test isapprox(LRMoE.expert_ll.(tmp, 0.0, 0, Inf, Inf), logcdf.(tmp, Inf) .+ log1mexp.(logcdf.(tmp, -1) .- logcdf.(tmp, Inf)), atol = 1e-6)
    @test isapprox(LRMoE.expert_ll.(tmp, 0.0, 0.75 .*x, 1.25 .*x, Inf), logcdf.(tmp, 1.25 .*x) .+ log1mexp.(logcdf.(tmp, ceil.(0.75 .*x) .- 1) .- logcdf.(tmp, 1.25 .*x)), atol = 1e-6)

    @test isapprox(LRMoE.expert_tn.(tmp, 0.0, x, x, Inf), fill(0.0, length(x)), atol = 1e-06)
    @test isapprox(LRMoE.expert_tn.(tmp, 0.0, 0.0, 0.0, 0.0), LRMoE.logpdf.(tmp, 0))
    @test isapprox(LRMoE.expert_tn.(tmp, 0.25 .*x, 0.75 .*x, 1.25 .*x, 1.5 .*x), logcdf.(tmp, 1.5 .*x) .+ log1mexp.(logcdf.(tmp, ceil.(0.25 .*x) .- 1) .- logcdf.(tmp, 1.5 .*x)), atol = 1e-6)

    # @test isapprox(LRMoE.expert_tn_bar.(tmp, 0.0, x, x, Inf), fill(-Inf, length(x)))
    @test isapprox(LRMoE.expert_tn_bar.(tmp, 0.0, 0.0, 0.0, 0.0), log1mexp(LRMoE.logpdf.(tmp, 0)), atol = 1e-06)
    @test isapprox(LRMoE.expert_tn_bar.(tmp, 0.25 .*x, 0.75 .*x, 1.25 .*x, 1.5 .*x), log1mexp.(logcdf.(tmp, 1.5 .*x) .+ log1mexp.(logcdf.(tmp, ceil.(0.25 .*x) .- 1) .- logcdf.(tmp, 1.5 .*x))), atol = 1e-6)

    # expo = rand(Distributions.Uniform(0.5, 5), 1)[1]
    # tmp_expo = LRMoE.GammaCountExpert(param1, param3/expo)

    # @test isapprox(LRMoE.expert_ll_exact.(tmp, x, exposure = expo), LRMoE.logpdf.(tmp_expo, x))

    # @test isapprox(LRMoE.expert_ll.(tmp, 0.0, x, x, Inf, exposure = expo), LRMoE.logpdf.(tmp_expo, x))
    # @test isapprox(LRMoE.expert_ll.(tmp, 0.0, 0.0, 0.0, 0.0, exposure = expo), LRMoE.logpdf.(tmp_expo, 0.0))
    # @test isapprox(LRMoE.expert_ll.(tmp, 0.0, x, Inf, Inf, exposure = expo), logcdf.(tmp_expo, Inf) .+ log1mexp.(logcdf.(tmp_expo, ceil.(x) .- 1) .- logcdf.(tmp_expo, Inf)), atol = 1e-6)
    # @test isapprox(LRMoE.expert_ll.(tmp, 0.0, 0, Inf, Inf, exposure = expo), logcdf.(tmp_expo, Inf) .+ log1mexp.(logcdf.(tmp_expo, -1) .- logcdf.(tmp_expo, Inf)), atol = 1e-6)
    # @test isapprox(LRMoE.expert_ll.(tmp, 0.0, 0.75 .*x, 1.25 .*x, Inf, exposure = expo), logcdf.(tmp_expo, 1.25 .*x) .+ log1mexp.(logcdf.(tmp_expo, ceil.(0.75 .*x) .- 1) .- logcdf.(tmp_expo, 1.25 .*x)), atol = 1e-6)

    # @test isapprox(LRMoE.expert_tn.(tmp, 0.0, x, x, Inf, exposure = expo), fill(0.0, length(x)), atol = 1e-06)
    # @test isapprox(LRMoE.expert_tn.(tmp, 0.0, 0.0, 0.0, 0.0, exposure = expo), LRMoE.logpdf.(tmp_expo, 0))
    # @test isapprox(LRMoE.expert_tn.(tmp, 0.25 .*x, 0.75 .*x, 1.25 .*x, 1.5 .*x, exposure = expo), logcdf.(tmp_expo, 1.5 .*x) .+ log1mexp.(logcdf.(tmp_expo, ceil.(0.25 .*x) .- 1) .- logcdf.(tmp_expo, 1.5 .*x)), atol = 1e-6)

    # # @test isapprox(LRMoE.expert_tn_bar.(tmp, 0.0, x, x, Inf, exposure = expo), fill(-Inf, length(x)))
    # @test isapprox(LRMoE.expert_tn_bar.(tmp, 0.0, 0.0, 0.0, 0.0, exposure = expo), log1mexp(LRMoE.logpdf.(tmp_expo, 0)), atol = 1e-06)
    # @test isapprox(LRMoE.expert_tn_bar.(tmp, 0.25 .*x, 0.75 .*x, 1.25 .*x, 1.5 .*x, exposure = expo), log1mexp.(logcdf.(tmp_expo, 1.5 .*x) .+ log1mexp.(logcdf.(tmp_expo, ceil.(0.25 .*x) .- 1) .- logcdf.(tmp_expo, 1.5 .*x))), atol = 1e-6)


    tmp = LRMoE.NegativeBinomialExpert(param1, param2)

    @test isapprox(LRMoE.expert_ll_exact.(tmp, x), LRMoE.logpdf.(tmp, x))

    @test isapprox(LRMoE.expert_ll.(tmp, 0.0, x, x, Inf), LRMoE.logpdf.(tmp, x))
    @test isapprox(LRMoE.expert_ll.(tmp, 0.0, 0.0, 0.0, 0.0), LRMoE.logpdf.(tmp, 0.0))
    @test isapprox(LRMoE.expert_ll.(tmp, 0.0, x, Inf, Inf), logcdf.(tmp, Inf) .+ log1mexp.(logcdf.(tmp, ceil.(x) .- 1) .- logcdf.(tmp, Inf)), atol = 1e-6)
    @test isapprox(LRMoE.expert_ll.(tmp, 0.0, 0, Inf, Inf), logcdf.(tmp, Inf) .+ log1mexp.(logcdf.(tmp, -1) .- logcdf.(tmp, Inf)), atol = 1e-6)
    @test isapprox(LRMoE.expert_ll.(tmp, 0.0, 0.75 .*x, 1.25 .*x, Inf), logcdf.(tmp, 1.25 .*x) .+ log1mexp.(logcdf.(tmp, ceil.(0.75 .*x) .- 1) .- logcdf.(tmp, 1.25 .*x)), atol = 1e-6)

    @test isapprox(LRMoE.expert_tn.(tmp, 0.0, x, x, Inf), fill(0.0, length(x)), atol = 1e-06)
    @test isapprox(LRMoE.expert_tn.(tmp, 0.0, 0.0, 0.0, 0.0), LRMoE.logpdf.(tmp, 0))
    @test isapprox(LRMoE.expert_tn.(tmp, 0.25 .*x, 0.75 .*x, 1.25 .*x, 1.5 .*x), logcdf.(tmp, 1.5 .*x) .+ log1mexp.(logcdf.(tmp, ceil.(0.25 .*x) .- 1) .- logcdf.(tmp, 1.5 .*x)), atol = 1e-6)

    # @test isapprox(LRMoE.expert_tn_bar.(tmp, 0.0, x, x, Inf), fill(-Inf, length(x)))
    @test isapprox(LRMoE.expert_tn_bar.(tmp, 0.0, 0.0, 0.0, 0.0), log1mexp(LRMoE.logpdf.(tmp, 0)), atol = 1e-06)
    @test isapprox(LRMoE.expert_tn_bar.(tmp, 0.25 .*x, 0.75 .*x, 1.25 .*x, 1.5 .*x), log1mexp.(logcdf.(tmp, 1.5 .*x) .+ log1mexp.(logcdf.(tmp, ceil.(0.25 .*x) .- 1) .- logcdf.(tmp, 1.5 .*x))), atol = 1e-6)

    # expo = rand(Distributions.Uniform(0.5, 5), 1)[1]
    # tmp_expo = LRMoE.NegativeBinomialExpert(param1*expo, param2)

    # @test isapprox(LRMoE.expert_ll_exact.(tmp, x, exposure = expo), LRMoE.logpdf.(tmp_expo, x))

    # @test isapprox(LRMoE.expert_ll.(tmp, 0.0, x, x, Inf, exposure = expo), LRMoE.logpdf.(tmp_expo, x))
    # @test isapprox(LRMoE.expert_ll.(tmp, 0.0, 0.0, 0.0, 0.0, exposure = expo), LRMoE.logpdf.(tmp_expo, 0.0))
    # @test isapprox(LRMoE.expert_ll.(tmp, 0.0, x, Inf, Inf, exposure = expo), logcdf.(tmp_expo, Inf) .+ log1mexp.(logcdf.(tmp_expo, ceil.(x) .- 1) .- logcdf.(tmp_expo, Inf)), atol = 1e-6)
    # @test isapprox(LRMoE.expert_ll.(tmp, 0.0, 0, Inf, Inf, exposure = expo), logcdf.(tmp_expo, Inf) .+ log1mexp.(logcdf.(tmp_expo, -1) .- logcdf.(tmp_expo, Inf)), atol = 1e-6)
    # @test isapprox(LRMoE.expert_ll.(tmp, 0.0, 0.75 .*x, 1.25 .*x, Inf, exposure = expo), logcdf.(tmp_expo, 1.25 .*x) .+ log1mexp.(logcdf.(tmp_expo, ceil.(0.75 .*x) .- 1) .- logcdf.(tmp_expo, 1.25 .*x)), atol = 1e-6)

    # @test isapprox(LRMoE.expert_tn.(tmp, 0.0, x, x, Inf, exposure = expo), fill(0.0, length(x)), atol = 1e-06)
    # @test isapprox(LRMoE.expert_tn.(tmp, 0.0, 0.0, 0.0, 0.0, exposure = expo), LRMoE.logpdf.(tmp_expo, 0))
    # @test isapprox(LRMoE.expert_tn.(tmp, 0.25 .*x, 0.75 .*x, 1.25 .*x, 1.5 .*x, exposure = expo), logcdf.(tmp_expo, 1.5 .*x) .+ log1mexp.(logcdf.(tmp_expo, ceil.(0.25 .*x) .- 1) .- logcdf.(tmp_expo, 1.5 .*x)), atol = 1e-6)

    # # @test isapprox(LRMoE.expert_tn_bar.(tmp, 0.0, x, x, Inf, exposure = expo), fill(-Inf, length(x)))
    # @test isapprox(LRMoE.expert_tn_bar.(tmp, 0.0, 0.0, 0.0, 0.0, exposure = expo), log1mexp(LRMoE.logpdf.(tmp_expo, 0)), atol = 1e-06)
    # @test isapprox(LRMoE.expert_tn_bar.(tmp, 0.25 .*x, 0.75 .*x, 1.25 .*x, 1.5 .*x, exposure = expo), log1mexp.(logcdf.(tmp_expo, 1.5 .*x) .+ log1mexp.(logcdf.(tmp_expo, ceil.(0.25 .*x) .- 1) .- logcdf.(tmp_expo, 1.5 .*x))), atol = 1e-6)


    tmp = LRMoE.PoissonExpert(param1)

    @test isapprox(LRMoE.expert_ll_exact.(tmp, x), LRMoE.logpdf.(tmp, x))

    @test isapprox(LRMoE.expert_ll.(tmp, 0.0, x, x, Inf), LRMoE.logpdf.(tmp, x))
    @test isapprox(LRMoE.expert_ll.(tmp, 0.0, 0.0, 0.0, 0.0), LRMoE.logpdf.(tmp, 0.0))
    @test isapprox(LRMoE.expert_ll.(tmp, 0.0, x, Inf, Inf), logcdf.(tmp, Inf) .+ log1mexp.(logcdf.(tmp, ceil.(x) .- 1) .- logcdf.(tmp, Inf)), atol = 1e-6)
    @test isapprox(LRMoE.expert_ll.(tmp, 0.0, 0, Inf, Inf), logcdf.(tmp, Inf) .+ log1mexp.(logcdf.(tmp, -1) .- logcdf.(tmp, Inf)), atol = 1e-6)
    @test isapprox(LRMoE.expert_ll.(tmp, 0.0, 0.75 .*x, 1.25 .*x, Inf), logcdf.(tmp, 1.25 .*x) .+ log1mexp.(logcdf.(tmp, ceil.(0.75 .*x) .- 1) .- logcdf.(tmp, 1.25 .*x)), atol = 1e-6)

    @test isapprox(LRMoE.expert_tn.(tmp, 0.0, x, x, Inf), fill(0.0, length(x)), atol = 1e-06)
    @test isapprox(LRMoE.expert_tn.(tmp, 0.0, 0.0, 0.0, 0.0), LRMoE.logpdf.(tmp, 0))
    @test isapprox(LRMoE.expert_tn.(tmp, 0.25 .*x, 0.75 .*x, 1.25 .*x, 1.5 .*x), logcdf.(tmp, 1.5 .*x) .+ log1mexp.(logcdf.(tmp, ceil.(0.25 .*x) .- 1) .- logcdf.(tmp, 1.5 .*x)), atol = 1e-6)

    # @test isapprox(LRMoE.expert_tn_bar.(tmp, 0.0, x, x, Inf), fill(-Inf, length(x)))
    @test isapprox(LRMoE.expert_tn_bar.(tmp, 0.0, 0.0, 0.0, 0.0), log1mexp(LRMoE.logpdf.(tmp, 0)), atol = 1e-06)
    @test isapprox(LRMoE.expert_tn_bar.(tmp, 0.25 .*x, 0.75 .*x, 1.25 .*x, 1.5 .*x), log1mexp.(logcdf.(tmp, 1.5 .*x) .+ log1mexp.(logcdf.(tmp, ceil.(0.25 .*x) .- 1) .- logcdf.(tmp, 1.5 .*x))), atol = 1e-6)

    # expo = rand(Distributions.Uniform(0.5, 5), 1)[1]
    # tmp_expo = LRMoE.PoissonExpert(param1*expo)

    # @test isapprox(LRMoE.expert_ll_exact.(tmp, x, exposure = expo), LRMoE.logpdf.(tmp_expo, x))

    # @test isapprox(LRMoE.expert_ll.(tmp, 0.0, x, x, Inf, exposure = expo), LRMoE.logpdf.(tmp_expo, x))
    # @test isapprox(LRMoE.expert_ll.(tmp, 0.0, 0.0, 0.0, 0.0, exposure = expo), LRMoE.logpdf.(tmp_expo, 0.0))
    # @test isapprox(LRMoE.expert_ll.(tmp, 0.0, x, Inf, Inf, exposure = expo), logcdf.(tmp_expo, Inf) .+ log1mexp.(logcdf.(tmp_expo, ceil.(x) .- 1) .- logcdf.(tmp_expo, Inf)), atol = 1e-6)
    # @test isapprox(LRMoE.expert_ll.(tmp, 0.0, 0, Inf, Inf, exposure = expo), logcdf.(tmp_expo, Inf) .+ log1mexp.(logcdf.(tmp_expo, -1) .- logcdf.(tmp_expo, Inf)), atol = 1e-6)
    # @test isapprox(LRMoE.expert_ll.(tmp, 0.0, 0.75 .*x, 1.25 .*x, Inf, exposure = expo), logcdf.(tmp_expo, 1.25 .*x) .+ log1mexp.(logcdf.(tmp_expo, ceil.(0.75 .*x) .- 1) .- logcdf.(tmp_expo, 1.25 .*x)), atol = 1e-6)

    # @test isapprox(LRMoE.expert_tn.(tmp, 0.0, x, x, Inf, exposure = expo), fill(0.0, length(x)), atol = 1e-06)
    # @test isapprox(LRMoE.expert_tn.(tmp, 0.0, 0.0, 0.0, 0.0, exposure = expo), LRMoE.logpdf.(tmp_expo, 0))
    # @test isapprox(LRMoE.expert_tn.(tmp, 0.25 .*x, 0.75 .*x, 1.25 .*x, 1.5 .*x, exposure = expo), logcdf.(tmp_expo, 1.5 .*x) .+ log1mexp.(logcdf.(tmp_expo, ceil.(0.25 .*x) .- 1) .- logcdf.(tmp_expo, 1.5 .*x)), atol = 1e-6)

    # # @test isapprox(LRMoE.expert_tn_bar.(tmp, 0.0, x, x, Inf, exposure = expo), fill(-Inf, length(x)))
    # @test isapprox(LRMoE.expert_tn_bar.(tmp, 0.0, 0.0, 0.0, 0.0, exposure = expo), log1mexp(LRMoE.logpdf.(tmp_expo, 0)), atol = 1e-06)
    # @test isapprox(LRMoE.expert_tn_bar.(tmp, 0.25 .*x, 0.75 .*x, 1.25 .*x, 1.5 .*x, exposure = expo), log1mexp.(logcdf.(tmp_expo, 1.5 .*x) .+ log1mexp.(logcdf.(tmp_expo, ceil.(0.25 .*x) .- 1) .- logcdf.(tmp_expo, 1.5 .*x))), atol = 1e-6)

end



@testset "expert_ll: discrete, ZI" begin

    param0 = rand(Distributions.Uniform(0, 1), 1)[1]
    param1 = 20
    param2 = rand(Distributions.Uniform(0, 1), 1)[1]
    param3 = 5.0

    l = Distributions.Binomial(param1, param2)
    x = rand(l, 20) .+ 1

    tmp = LRMoE.ZIBinomialExpert(param0, param1, param2)

    @test isapprox(LRMoE.expert_ll_exact.(tmp, x), log.(1-param0) .+ LRMoE.logpdf.(tmp, x))

    @test isapprox(LRMoE.expert_ll.(tmp, 0.0, x, x, Inf), log.(1-param0) .+ LRMoE.logpdf.(tmp, x))
    @test isapprox(LRMoE.expert_ll.(tmp, 0.0, 0.0, 0.0, 0.0), log(param0 + (1-param0) .* exp.(LRMoE.logpdf.(tmp, 0.0))))
    @test isapprox(LRMoE.expert_ll.(tmp, 0.0, x, Inf, Inf), log.(1-param0) .+ logcdf.(tmp, Inf) .+ log1mexp.(logcdf.(tmp, ceil.(x) .- 1) .- logcdf.(tmp, Inf)), atol = 1e-6)
    @test isapprox(LRMoE.expert_ll.(tmp, 0.0, 0, Inf, Inf), logcdf.(tmp, Inf) .+ log1mexp.(logcdf.(tmp, -1) .- logcdf.(tmp, Inf)), atol = 1e-6)
    @test isapprox(LRMoE.expert_ll.(tmp, 0.0, 0.75 .*x, 1.25 .*x, Inf), log.(1-param0) .+ logcdf.(tmp, 1.25 .*x) .+ log1mexp.(logcdf.(tmp, ceil.(0.75 .*x) .- 1) .- logcdf.(tmp, 1.25 .*x)), atol = 1e-6)

    @test isapprox(LRMoE.expert_tn.(tmp, 0.0, x, x, Inf), fill(0.0, length(x)), atol = 1e-06)
    @test isapprox(LRMoE.expert_tn.(tmp, 0.0, 0.0, 0.0, 0.0), log(param0 + (1-param0) .* exp.(LRMoE.logpdf.(tmp, 0.0))))
    @test isapprox(LRMoE.expert_tn.(tmp, 0.25 .*x, 0.75 .*x, 1.25 .*x, 1.5 .*x), log.(1-param0) .+ logcdf.(tmp, 1.5 .*x) .+ log1mexp.(logcdf.(tmp, ceil.(0.25 .*x) .- 1) .- logcdf.(tmp, 1.5 .*x)), atol = 1e-6)

    # @test isapprox(LRMoE.expert_tn_bar.(tmp, 0.0, x, x, Inf), fill(-Inf, length(x)))
    @test isapprox(LRMoE.expert_tn_bar.(tmp, 0.0, 0.0, 0.0, 0.0), log1mexp(log(param0 + (1-param0) .* exp.(LRMoE.logpdf.(tmp, 0.0)))), atol = 1e-06)
    # @test isapprox(LRMoE.expert_tn_bar.(tmp, 0.25 .*x, 0.75 .*x, 1.25 .*x, 1.5 .*x), log1mexp.(logcdf.(tmp, 1.5 .*x) .+ log1mexp.(logcdf.(tmp, ceil.(0.25 .*x) .- 1) .- logcdf.(tmp, 1.5 .*x))), atol = 1e-6)

    tmp = LRMoE.ZIGammaCountExpert(param0, param1, param3)

    @test isapprox(LRMoE.expert_ll_exact.(tmp, x), log.(1-param0) .+ LRMoE.logpdf.(tmp, x))

    @test isapprox(LRMoE.expert_ll.(tmp, 0.0, x, x, Inf), log.(1-param0) .+ LRMoE.logpdf.(tmp, x))
    @test isapprox(LRMoE.expert_ll.(tmp, 0.0, 0.0, 0.0, 0.0), log(param0 + (1-param0) .* exp.(LRMoE.logpdf.(tmp, 0.0))))
    @test isapprox(LRMoE.expert_ll.(tmp, 0.0, x, Inf, Inf), log.(1-param0) .+ logcdf.(tmp, Inf) .+ log1mexp.(logcdf.(tmp, ceil.(x) .- 1) .- logcdf.(tmp, Inf)), atol = 1e-6)
    @test isapprox(LRMoE.expert_ll.(tmp, 0.0, 0, Inf, Inf), logcdf.(tmp, Inf) .+ log1mexp.(logcdf.(tmp, -1) .- logcdf.(tmp, Inf)), atol = 1e-6)
    @test isapprox(LRMoE.expert_ll.(tmp, 0.0, 0.75 .*x, 1.25 .*x, Inf), log.(1-param0) .+ logcdf.(tmp, 1.25 .*x) .+ log1mexp.(logcdf.(tmp, ceil.(0.75 .*x) .- 1) .- logcdf.(tmp, 1.25 .*x)), atol = 1e-6)

    @test isapprox(LRMoE.expert_tn.(tmp, 0.0, x, x, Inf), fill(0.0, length(x)), atol = 1e-06)
    @test isapprox(LRMoE.expert_tn.(tmp, 0.0, 0.0, 0.0, 0.0), log(param0 + (1-param0) .* exp.(LRMoE.logpdf.(tmp, 0.0))))
    @test isapprox(LRMoE.expert_tn.(tmp, 0.25 .*x, 0.75 .*x, 1.25 .*x, 1.5 .*x), log.(1-param0) .+ logcdf.(tmp, 1.5 .*x) .+ log1mexp.(logcdf.(tmp, ceil.(0.25 .*x) .- 1) .- logcdf.(tmp, 1.5 .*x)), atol = 1e-6)

    # @test isapprox(LRMoE.expert_tn_bar.(tmp, 0.0, x, x, Inf), fill(-Inf, length(x)))
    @test isapprox(LRMoE.expert_tn_bar.(tmp, 0.0, 0.0, 0.0, 0.0), log1mexp(log(param0 + (1-param0) .* exp.(LRMoE.logpdf.(tmp, 0.0)))), atol = 1e-06)
    # @test isapprox(LRMoE.expert_tn_bar.(tmp, 0.25 .*x, 0.75 .*x, 1.25 .*x, 1.5 .*x), log1mexp.(logcdf.(tmp, 1.5 .*x) .+ log1mexp.(logcdf.(tmp, ceil.(0.25 .*x) .- 1) .- logcdf.(tmp, 1.5 .*x))), atol = 1e-6)

    # expo = rand(Distributions.Uniform(0.5, 5), 1)[1]
    # tmp_expo = LRMoE.ZIGammaCountExpert(param0, param1, param3/expo)

    # @test isapprox(LRMoE.expert_ll_exact.(tmp, x, exposure = expo), log.(1-param0) .+ LRMoE.logpdf.(tmp_expo, x))

    # @test isapprox(LRMoE.expert_ll.(tmp, 0.0, x, x, Inf, exposure = expo), log.(1-param0) .+ LRMoE.logpdf.(tmp_expo, x))
    # @test isapprox(LRMoE.expert_ll.(tmp, 0.0, 0.0, 0.0, 0.0, exposure = expo), log(param0 + (1-param0) .* exp.(LRMoE.logpdf.(tmp_expo, 0.0))))
    # @test isapprox(LRMoE.expert_ll.(tmp, 0.0, x, Inf, Inf, exposure = expo), log.(1-param0) .+ logcdf.(tmp_expo, Inf) .+ log1mexp.(logcdf.(tmp_expo, ceil.(x) .- 1) .- logcdf.(tmp_expo, Inf)), atol = 1e-6)
    # @test isapprox(LRMoE.expert_ll.(tmp, 0.0, 0, Inf, Inf, exposure = expo), logcdf.(tmp_expo, Inf) .+ log1mexp.(logcdf.(tmp_expo, -1) .- logcdf.(tmp_expo, Inf)), atol = 1e-6)
    # @test isapprox(LRMoE.expert_ll.(tmp, 0.0, 0.75 .*x, 1.25 .*x, Inf, exposure = expo), log.(1-param0) .+ logcdf.(tmp_expo, 1.25 .*x) .+ log1mexp.(logcdf.(tmp_expo, ceil.(0.75 .*x) .- 1) .- logcdf.(tmp_expo, 1.25 .*x)), atol = 1e-6)

    # @test isapprox(LRMoE.expert_tn.(tmp, 0.0, x, x, Inf, exposure = expo), fill(0.0, length(x)), atol = 1e-06)
    # @test isapprox(LRMoE.expert_tn.(tmp, 0.0, 0.0, 0.0, 0.0, exposure = expo), log(param0 + (1-param0) .* exp.(LRMoE.logpdf.(tmp_expo, 0.0))))
    # @test isapprox(LRMoE.expert_tn.(tmp, 0.25 .*x, 0.75 .*x, 1.25 .*x, 1.5 .*x, exposure = expo), log.(1-param0) .+ logcdf.(tmp_expo, 1.5 .*x) .+ log1mexp.(logcdf.(tmp_expo, ceil.(0.25 .*x) .- 1) .- logcdf.(tmp_expo, 1.5 .*x)), atol = 1e-6)

    # # @test isapprox(LRMoE.expert_tn_bar.(tmp, 0.0, x, x, Inf, exposure = expo), fill(-Inf, length(x)))
    # @test isapprox(LRMoE.expert_tn_bar.(tmp, 0.0, 0.0, 0.0, 0.0, exposure = expo), log1mexp(log(param0 + (1-param0) .* exp.(LRMoE.logpdf.(tmp_expo, 0.0)))), atol = 1e-06)
    # # @test isapprox(LRMoE.expert_tn_bar.(tmp, 0.25 .*x, 0.75 .*x, 1.25 .*x, 1.5 .*x, exposure = expo), log1mexp.(logcdf.(tmp_expo, 1.5 .*x) .+ log1mexp.(logcdf.(tmp_expo, ceil.(0.25 .*x) .- 1) .- logcdf.(tmp_expo, 1.5 .*x))), atol = 1e-6)


    tmp = LRMoE.ZINegativeBinomialExpert(param0, param1, param2)

    @test isapprox(LRMoE.expert_ll_exact.(tmp, x), log.(1-param0) .+ LRMoE.logpdf.(tmp, x))

    @test isapprox(LRMoE.expert_ll.(tmp, 0.0, x, x, Inf), log.(1-param0) .+ LRMoE.logpdf.(tmp, x))
    @test isapprox(LRMoE.expert_ll.(tmp, 0.0, 0.0, 0.0, 0.0), log(param0 + (1-param0) .* exp.(LRMoE.logpdf.(tmp, 0.0))))
    @test isapprox(LRMoE.expert_ll.(tmp, 0.0, x, Inf, Inf), log.(1-param0) .+ logcdf.(tmp, Inf) .+ log1mexp.(logcdf.(tmp, ceil.(x) .- 1) .- logcdf.(tmp, Inf)), atol = 1e-6)
    @test isapprox(LRMoE.expert_ll.(tmp, 0.0, 0, Inf, Inf), logcdf.(tmp, Inf) .+ log1mexp.(logcdf.(tmp, -1) .- logcdf.(tmp, Inf)), atol = 1e-6)
    @test isapprox(LRMoE.expert_ll.(tmp, 0.0, 0.75 .*x, 1.25 .*x, Inf), log.(1-param0) .+ logcdf.(tmp, 1.25 .*x) .+ log1mexp.(logcdf.(tmp, ceil.(0.75 .*x) .- 1) .- logcdf.(tmp, 1.25 .*x)), atol = 1e-6)

    @test isapprox(LRMoE.expert_tn.(tmp, 0.0, x, x, Inf), fill(0.0, length(x)), atol = 1e-06)
    @test isapprox(LRMoE.expert_tn.(tmp, 0.0, 0.0, 0.0, 0.0), log(param0 + (1-param0) .* exp.(LRMoE.logpdf.(tmp, 0.0))))
    @test isapprox(LRMoE.expert_tn.(tmp, 0.25 .*x, 0.75 .*x, 1.25 .*x, 1.5 .*x), log.(1-param0) .+ logcdf.(tmp, 1.5 .*x) .+ log1mexp.(logcdf.(tmp, ceil.(0.25 .*x) .- 1) .- logcdf.(tmp, 1.5 .*x)), atol = 1e-6)

    # @test isapprox(LRMoE.expert_tn_bar.(tmp, 0.0, x, x, Inf), fill(-Inf, length(x)))
    @test isapprox(LRMoE.expert_tn_bar.(tmp, 0.0, 0.0, 0.0, 0.0), log1mexp(log(param0 + (1-param0) .* exp.(LRMoE.logpdf.(tmp, 0.0)))), atol = 1e-06)
    # @test isapprox(LRMoE.expert_tn_bar.(tmp, 0.25 .*x, 0.75 .*x, 1.25 .*x, 1.5 .*x), log1mexp.(logcdf.(tmp, 1.5 .*x) .+ log1mexp.(logcdf.(tmp, ceil.(0.25 .*x) .- 1) .- logcdf.(tmp, 1.5 .*x))), atol = 1e-6)

    # expo = rand(Distributions.Uniform(0.5, 5), 1)[1]
    # tmp_expo = LRMoE.ZINegativeBinomialExpert(param0, param1*expo, param2)

    # @test isapprox(LRMoE.expert_ll_exact.(tmp, x, exposure = expo), log.(1-param0) .+ LRMoE.logpdf.(tmp_expo, x))

    # @test isapprox(LRMoE.expert_ll.(tmp, 0.0, x, x, Inf, exposure = expo), log.(1-param0) .+ LRMoE.logpdf.(tmp_expo, x))
    # @test isapprox(LRMoE.expert_ll.(tmp, 0.0, 0.0, 0.0, 0.0, exposure = expo), log(param0 + (1-param0) .* exp.(LRMoE.logpdf.(tmp_expo, 0.0))))
    # @test isapprox(LRMoE.expert_ll.(tmp, 0.0, x, Inf, Inf, exposure = expo), log.(1-param0) .+ logcdf.(tmp_expo, Inf) .+ log1mexp.(logcdf.(tmp_expo, ceil.(x) .- 1) .- logcdf.(tmp_expo, Inf)), atol = 1e-6)
    # @test isapprox(LRMoE.expert_ll.(tmp, 0.0, 0, Inf, Inf, exposure = expo), logcdf.(tmp_expo, Inf) .+ log1mexp.(logcdf.(tmp_expo, -1) .- logcdf.(tmp_expo, Inf)), atol = 1e-6)
    # @test isapprox(LRMoE.expert_ll.(tmp, 0.0, 0.75 .*x, 1.25 .*x, Inf, exposure = expo), log.(1-param0) .+ logcdf.(tmp_expo, 1.25 .*x) .+ log1mexp.(logcdf.(tmp_expo, ceil.(0.75 .*x) .- 1) .- logcdf.(tmp_expo, 1.25 .*x)), atol = 1e-6)

    # @test isapprox(LRMoE.expert_tn.(tmp, 0.0, x, x, Inf, exposure = expo), fill(0.0, length(x)), atol = 1e-06)
    # @test isapprox(LRMoE.expert_tn.(tmp, 0.0, 0.0, 0.0, 0.0, exposure = expo), log(param0 + (1-param0) .* exp.(LRMoE.logpdf.(tmp_expo, 0.0))))
    # @test isapprox(LRMoE.expert_tn.(tmp, 0.25 .*x, 0.75 .*x, 1.25 .*x, 1.5 .*x, exposure = expo), log.(1-param0) .+ logcdf.(tmp_expo, 1.5 .*x) .+ log1mexp.(logcdf.(tmp_expo, ceil.(0.25 .*x) .- 1) .- logcdf.(tmp_expo, 1.5 .*x)), atol = 1e-6)

    # # @test isapprox(LRMoE.expert_tn_bar.(tmp, 0.0, x, x, Inf, exposure = expo), fill(-Inf, length(x)))
    # @test isapprox(LRMoE.expert_tn_bar.(tmp, 0.0, 0.0, 0.0, 0.0, exposure = expo), log1mexp(log(param0 + (1-param0) .* exp.(LRMoE.logpdf.(tmp_expo, 0.0)))), atol = 1e-06)
    # # @test isapprox(LRMoE.expert_tn_bar.(tmp, 0.25 .*x, 0.75 .*x, 1.25 .*x, 1.5 .*x, exposure = expo), log1mexp.(logcdf.(tmp_expo, 1.5 .*x) .+ log1mexp.(logcdf.(tmp_expo, ceil.(0.25 .*x) .- 1) .- logcdf.(tmp_expo, 1.5 .*x))), atol = 1e-6)


    tmp = LRMoE.ZIPoissonExpert(param0, param1)

    @test isapprox(LRMoE.expert_ll_exact.(tmp, x), log.(1-param0) .+ LRMoE.logpdf.(tmp, x))

    @test isapprox(LRMoE.expert_ll.(tmp, 0.0, x, x, Inf), log.(1-param0) .+ LRMoE.logpdf.(tmp, x))
    @test isapprox(LRMoE.expert_ll.(tmp, 0.0, 0.0, 0.0, 0.0), log(param0 + (1-param0) .* exp.(LRMoE.logpdf.(tmp, 0.0))))
    @test isapprox(LRMoE.expert_ll.(tmp, 0.0, x, Inf, Inf), log.(1-param0) .+ logcdf.(tmp, Inf) .+ log1mexp.(logcdf.(tmp, ceil.(x) .- 1) .- logcdf.(tmp, Inf)), atol = 1e-6)
    @test isapprox(LRMoE.expert_ll.(tmp, 0.0, 0, Inf, Inf), logcdf.(tmp, Inf) .+ log1mexp.(logcdf.(tmp, -1) .- logcdf.(tmp, Inf)), atol = 1e-6)
    @test isapprox(LRMoE.expert_ll.(tmp, 0.0, 0.75 .*x, 1.25 .*x, Inf), log.(1-param0) .+ logcdf.(tmp, 1.25 .*x) .+ log1mexp.(logcdf.(tmp, ceil.(0.75 .*x) .- 1) .- logcdf.(tmp, 1.25 .*x)), atol = 1e-6)

    @test isapprox(LRMoE.expert_tn.(tmp, 0.0, x, x, Inf), fill(0.0, length(x)), atol = 1e-06)
    @test isapprox(LRMoE.expert_tn.(tmp, 0.0, 0.0, 0.0, 0.0), log(param0 + (1-param0) .* exp.(LRMoE.logpdf.(tmp, 0.0))))
    @test isapprox(LRMoE.expert_tn.(tmp, 0.25 .*x, 0.75 .*x, 1.25 .*x, 1.5 .*x), log.(1-param0) .+ logcdf.(tmp, 1.5 .*x) .+ log1mexp.(logcdf.(tmp, ceil.(0.25 .*x) .- 1) .- logcdf.(tmp, 1.5 .*x)), atol = 1e-6)

    # @test isapprox(LRMoE.expert_tn_bar.(tmp, 0.0, x, x, Inf), fill(-Inf, length(x)))
    @test isapprox(LRMoE.expert_tn_bar.(tmp, 0.0, 0.0, 0.0, 0.0), log1mexp(log(param0 + (1-param0) .* exp.(LRMoE.logpdf.(tmp, 0.0)))), atol = 1e-06)
    # @test isapprox(LRMoE.expert_tn_bar.(tmp, 0.25 .*x, 0.75 .*x, 1.25 .*x, 1.5 .*x), log1mexp.(logcdf.(tmp, 1.5 .*x) .+ log1mexp.(logcdf.(tmp, ceil.(0.25 .*x) .- 1) .- logcdf.(tmp, 1.5 .*x))), atol = 1e-6)

    # expo = rand(Distributions.Uniform(0.5, 5), 1)[1]
    # tmp_expo = LRMoE.ZIPoissonExpert(param0, param1*expo)

    # @test isapprox(LRMoE.expert_ll_exact.(tmp, x, exposure = expo), log.(1-param0) .+ LRMoE.logpdf.(tmp_expo, x))

    # @test isapprox(LRMoE.expert_ll.(tmp, 0.0, x, x, Inf, exposure = expo), log.(1-param0) .+ LRMoE.logpdf.(tmp_expo, x))
    # @test isapprox(LRMoE.expert_ll.(tmp, 0.0, 0.0, 0.0, 0.0, exposure = expo), log(param0 + (1-param0) .* exp.(LRMoE.logpdf.(tmp_expo, 0.0))))
    # @test isapprox(LRMoE.expert_ll.(tmp, 0.0, x, Inf, Inf, exposure = expo), log.(1-param0) .+ logcdf.(tmp_expo, Inf) .+ log1mexp.(logcdf.(tmp_expo, ceil.(x) .- 1) .- logcdf.(tmp_expo, Inf)), atol = 1e-6)
    # @test isapprox(LRMoE.expert_ll.(tmp, 0.0, 0, Inf, Inf, exposure = expo), logcdf.(tmp_expo, Inf) .+ log1mexp.(logcdf.(tmp_expo, -1) .- logcdf.(tmp_expo, Inf)), atol = 1e-6)
    # @test isapprox(LRMoE.expert_ll.(tmp, 0.0, 0.75 .*x, 1.25 .*x, Inf, exposure = expo), log.(1-param0) .+ logcdf.(tmp_expo, 1.25 .*x) .+ log1mexp.(logcdf.(tmp_expo, ceil.(0.75 .*x) .- 1) .- logcdf.(tmp_expo, 1.25 .*x)), atol = 1e-6)

    # @test isapprox(LRMoE.expert_tn.(tmp, 0.0, x, x, Inf, exposure = expo), fill(0.0, length(x)), atol = 1e-06)
    # @test isapprox(LRMoE.expert_tn.(tmp, 0.0, 0.0, 0.0, 0.0, exposure = expo), log(param0 + (1-param0) .* exp.(LRMoE.logpdf.(tmp_expo, 0.0))))
    # @test isapprox(LRMoE.expert_tn.(tmp, 0.25 .*x, 0.75 .*x, 1.25 .*x, 1.5 .*x, exposure = expo), log.(1-param0) .+ logcdf.(tmp_expo, 1.5 .*x) .+ log1mexp.(logcdf.(tmp_expo, ceil.(0.25 .*x) .- 1) .- logcdf.(tmp_expo, 1.5 .*x)), atol = 1e-6)

    # # @test isapprox(LRMoE.expert_tn_bar.(tmp, 0.0, x, x, Inf, exposure = expo), fill(-Inf, length(x)))
    # @test isapprox(LRMoE.expert_tn_bar.(tmp, 0.0, 0.0, 0.0, 0.0, exposure = expo), log1mexp(log(param0 + (1-param0) .* exp.(LRMoE.logpdf.(tmp_expo, 0.0)))), atol = 1e-06)
    # # @test isapprox(LRMoE.expert_tn_bar.(tmp, 0.25 .*x, 0.75 .*x, 1.25 .*x, 1.5 .*x, exposure = expo), log1mexp.(logcdf.(tmp_expo, 1.5 .*x) .+ log1mexp.(logcdf.(tmp_expo, ceil.(0.25 .*x) .- 1) .- logcdf.(tmp_expo, 1.5 .*x))), atol = 1e-6)

end



@testset "exposurize_expert" begin

    param0 = rand(Distributions.Uniform(0, 1), 1)[1]
    param1 = rand(Distributions.Uniform(10, 20), 1)[1]
    param2 = rand(Distributions.Uniform(0, 1), 1)[1]
    param3 = rand(Distributions.Uniform(5, 10), 1)[1]

    expo = rand(Distributions.Uniform(0.5, 20), 1)[1]

    tmp = LRMoE.BurrExpert(param1, param2, param3)
    @test tmp == LRMoE.exposurize_expert(tmp, exposure = expo)
    tmp = LRMoE.ZIBurrExpert(param0, param1, param2, param3)
    @test tmp == LRMoE.exposurize_expert(tmp, exposure = expo)

    tmp = LRMoE.GammaExpert(param1, param2)
    @test tmp == LRMoE.exposurize_expert(tmp, exposure = expo)
    tmp = LRMoE.ZIGammaExpert(param0, param1, param2)
    @test tmp == LRMoE.exposurize_expert(tmp, exposure = expo)

    tmp = LRMoE.InverseGaussianExpert(param1, param2)
    @test tmp == LRMoE.exposurize_expert(tmp, exposure = expo)
    tmp = LRMoE.ZIInverseGaussianExpert(param0, param1, param2)
    @test tmp == LRMoE.exposurize_expert(tmp, exposure = expo)

    tmp = LRMoE.LogNormalExpert(param1, param2)
    @test tmp == LRMoE.exposurize_expert(tmp, exposure = expo)
    tmp = LRMoE.ZILogNormalExpert(param0, param1, param2)
    @test tmp == LRMoE.exposurize_expert(tmp, exposure = expo)

    tmp = LRMoE.WeibullExpert(param1, param2)
    @test tmp == LRMoE.exposurize_expert(tmp, exposure = expo)
    tmp = LRMoE.ZIWeibullExpert(param0, param1, param2)
    @test tmp == LRMoE.exposurize_expert(tmp, exposure = expo)


    tmp = LRMoE.BinomialExpert(20, param0)
    @test tmp == LRMoE.exposurize_expert(tmp, exposure = expo)
    tmp = LRMoE.ZIBinomialExpert(0.20, 20, param0)
    @test tmp == LRMoE.exposurize_expert(tmp, exposure = expo)


    tmp = LRMoE.GammaCountExpert(param1, param2)
    @test tmp != LRMoE.exposurize_expert(tmp, exposure = expo)
    @test isapprox([LRMoE.params(tmp)...] .* [1, 1/expo], [LRMoE.params(LRMoE.exposurize_expert(tmp, exposure = expo))...], atol = 1e-6)
    tmp = LRMoE.ZIGammaCountExpert(param0, param1, param2)
    @test tmp != LRMoE.exposurize_expert(tmp, exposure = expo)
    @test isapprox([LRMoE.params(tmp)...] .* [1, 1, 1/expo], [LRMoE.params(LRMoE.exposurize_expert(tmp, exposure = expo))...], atol = 1e-6)

    tmp = LRMoE.NegativeBinomialExpert(param1, param0)
    @test tmp != LRMoE.exposurize_expert(tmp, exposure = expo)
    @test [LRMoE.params(tmp)...] .* [expo, 1] == [LRMoE.params(LRMoE.exposurize_expert(tmp, exposure = expo))...]
    tmp = LRMoE.ZINegativeBinomialExpert(param0, param1, param0)
    @test tmp != LRMoE.exposurize_expert(tmp, exposure = expo)
    @test [LRMoE.params(tmp)...] .* [1, expo, 1] == [LRMoE.params(LRMoE.exposurize_expert(tmp, exposure = expo))...]

    tmp = LRMoE.PoissonExpert(param1)
    @test tmp != LRMoE.exposurize_expert(tmp, exposure = expo)
    @test [LRMoE.params(tmp)...] .* [expo] == [LRMoE.params(LRMoE.exposurize_expert(tmp, exposure = expo))...]
    tmp = LRMoE.ZIPoissonExpert(param0, param1)
    @test tmp != LRMoE.exposurize_expert(tmp, exposure = expo)
    @test [LRMoE.params(tmp)...] .* [1, expo] == [LRMoE.params(LRMoE.exposurize_expert(tmp, exposure = expo))...]


    

    model = [LogNormalExpert(param1, param2) ZIGammaExpert(param0, param1, param2) WeibullExpert(param1, param2) BurrExpert(param1, param2, param3);
            BinomialExpert(20, param0) PoissonExpert(param1) ZIGammaCountExpert(param0, param1, param2) ZINegativeBinomialExpert(param0, param1, param0)]

    expo = rand(Distributions.Uniform(0.5, 20), 15)

    @test size(LRMoE.exposurize_model(model, exposure = expo)) == (2, 4, 15)

    
    @test mean.(LRMoE.exposurize_model(model, exposure = expo))[1,1,:] == fill(mean(model[1,1]), 15)
    @test mean.(LRMoE.exposurize_model(model, exposure = expo))[1,2,:] == fill(mean(model[1,2]), 15)
    @test mean.(LRMoE.exposurize_model(model, exposure = expo))[1,3,:] == fill(mean(model[1,3]), 15)
    @test mean.(LRMoE.exposurize_model(model, exposure = expo))[1,4,:] == fill(mean(model[1,4]), 15)
    
    @test mean.(LRMoE.exposurize_model(model, exposure = expo))[2,1,:] == fill(mean(model[2,1]), 15)
    @test mean.(LRMoE.exposurize_model(model, exposure = expo))[2,2,:] == mean(model[2,2]) .* expo


    model_expod = LRMoE.exposurize_model(model, exposure = expo)
    Y_tmp = hcat(rand(Distributions.LogNormal(2, 1), 15), rand(Distributions.Poisson(5), 15))

    @test size(LRMoE.expert_ll_ind_mat_exact(Y_tmp[1,:], model_expod[:,:,1])) == (2, 4)
    @test size(LRMoE.expert_ll_ind_mat_exact(Y_tmp[5,:], model_expod[:,:,5])) == (2, 4)
    @test size(LRMoE.expert_ll_ind_mat_exact(Y_tmp[12,:], model_expod[:,:,12])) == (2, 4)
    @test size(LRMoE.expert_ll_list_exact(Y_tmp, model_expod)) == (2, 4, 15)

    idx = rand(1:15, 1)[1]
    mat_tmp = LRMoE.expert_ll_ind_mat_exact(Y_tmp[idx,:], model_expod[:,:,idx])
    @test mat_tmp[1,1] == LRMoE.expert_ll_exact(model_expod[1,1,idx], Y_tmp[idx,1])
    @test mat_tmp[1,2] == LRMoE.expert_ll_exact(model_expod[1,2,idx], Y_tmp[idx,1])
    @test mat_tmp[1,3] == LRMoE.expert_ll_exact(model_expod[1,3,idx], Y_tmp[idx,1])
    @test mat_tmp[1,4] == LRMoE.expert_ll_exact(model_expod[1,4,idx], Y_tmp[idx,1])
    @test mat_tmp[2,1] == LRMoE.expert_ll_exact(model_expod[2,1,idx], Y_tmp[idx,2])
    @test mat_tmp[2,2] == LRMoE.expert_ll_exact(model_expod[2,2,idx], Y_tmp[idx,2])
    @test mat_tmp[2,3] == LRMoE.expert_ll_exact(model_expod[2,3,idx], Y_tmp[idx,2])
    @test mat_tmp[2,3] == LRMoE.expert_ll_exact(model_expod[2,3,idx], Y_tmp[idx,2])
    
    idx = rand(1:15, 1)[1]
    cube_tmp = LRMoE.expert_ll_list_exact(Y_tmp, model_expod)
    @test cube_tmp[1,1,idx] == LRMoE.expert_ll_exact(model_expod[1,1,idx], Y_tmp[idx,1])
    @test cube_tmp[1,2,idx] == LRMoE.expert_ll_exact(model_expod[1,2,idx], Y_tmp[idx,1])
    @test cube_tmp[1,3,idx] == LRMoE.expert_ll_exact(model_expod[1,3,idx], Y_tmp[idx,1])
    @test cube_tmp[1,4,idx] == LRMoE.expert_ll_exact(model_expod[1,4,idx], Y_tmp[idx,1])
    @test cube_tmp[2,1,idx] == LRMoE.expert_ll_exact(model_expod[2,1,idx], Y_tmp[idx,2])
    @test cube_tmp[2,2,idx] == LRMoE.expert_ll_exact(model_expod[2,2,idx], Y_tmp[idx,2])
    @test cube_tmp[2,3,idx] == LRMoE.expert_ll_exact(model_expod[2,3,idx], Y_tmp[idx,2])
    @test cube_tmp[2,4,idx] == LRMoE.expert_ll_exact(model_expod[2,4,idx], Y_tmp[idx,2])


end



@testset "exposurize_model" begin
    param0 = rand(Distributions.Uniform(0, 1), 1)[1]
    param1 = rand(Distributions.Uniform(10, 20), 1)[1]
    param2 = rand(Distributions.Uniform(0, 1), 1)[1]
    param3 = rand(Distributions.Uniform(5, 10), 1)[1]

    model = [LogNormalExpert(param1, param2) ZIGammaExpert(param0, param1, param2) WeibullExpert(param1, param2) BurrExpert(param1, param2, param3);
            BinomialExpert(20, param0) PoissonExpert(param1) ZIGammaCountExpert(param0, param1, param2) ZINegativeBinomialExpert(param0, param1, param0)]

    expo = rand(Distributions.Uniform(0.5, 20), 15)

    @test size(LRMoE.exposurize_model(model, exposure = expo)) == (2, 4, 15)

    
    @test mean.(LRMoE.exposurize_model(model, exposure = expo))[1,1,:] == fill(mean(model[1,1]), 15)
    @test mean.(LRMoE.exposurize_model(model, exposure = expo))[1,2,:] == fill(mean(model[1,2]), 15)
    @test mean.(LRMoE.exposurize_model(model, exposure = expo))[1,3,:] == fill(mean(model[1,3]), 15)
    @test mean.(LRMoE.exposurize_model(model, exposure = expo))[1,4,:] == fill(mean(model[1,4]), 15)
    
    @test mean.(LRMoE.exposurize_model(model, exposure = expo))[2,1,:] == fill(mean(model[2,1]), 15)
    @test mean.(LRMoE.exposurize_model(model, exposure = expo))[2,2,:] == mean(model[2,2]) .* expo


    model_expod = LRMoE.exposurize_model(model, exposure = expo)
    Y_tmp = hcat(rand(Distributions.LogNormal(2, 1), 15), rand(Distributions.Poisson(5), 15))

    @test size(LRMoE.expert_ll_ind_mat_exact(Y_tmp[1,:], model_expod[:,:,1])) == (2, 4)
    @test size(LRMoE.expert_ll_ind_mat_exact(Y_tmp[5,:], model_expod[:,:,5])) == (2, 4)
    @test size(LRMoE.expert_ll_ind_mat_exact(Y_tmp[12,:], model_expod[:,:,12])) == (2, 4)
    @test size(LRMoE.expert_ll_list_exact(Y_tmp, model_expod)) == (2, 4, 15)

    idx = rand(1:15, 1)[1]
    mat_tmp = LRMoE.expert_ll_ind_mat_exact(Y_tmp[idx,:], model_expod[:,:,idx])
    @test mat_tmp[1,1] == LRMoE.expert_ll_exact(model_expod[1,1,idx], Y_tmp[idx,1])
    @test mat_tmp[1,2] == LRMoE.expert_ll_exact(model_expod[1,2,idx], Y_tmp[idx,1])
    @test mat_tmp[1,3] == LRMoE.expert_ll_exact(model_expod[1,3,idx], Y_tmp[idx,1])
    @test mat_tmp[1,4] == LRMoE.expert_ll_exact(model_expod[1,4,idx], Y_tmp[idx,1])
    @test mat_tmp[2,1] == LRMoE.expert_ll_exact(model_expod[2,1,idx], Y_tmp[idx,2])
    @test mat_tmp[2,2] == LRMoE.expert_ll_exact(model_expod[2,2,idx], Y_tmp[idx,2])
    @test mat_tmp[2,3] == LRMoE.expert_ll_exact(model_expod[2,3,idx], Y_tmp[idx,2])
    @test mat_tmp[2,3] == LRMoE.expert_ll_exact(model_expod[2,3,idx], Y_tmp[idx,2])
    
    idx = rand(1:15, 1)[1]
    cube_tmp = LRMoE.expert_ll_list_exact(Y_tmp, model_expod)
    @test cube_tmp[1,1,idx] == LRMoE.expert_ll_exact(model_expod[1,1,idx], Y_tmp[idx,1])
    @test cube_tmp[1,2,idx] == LRMoE.expert_ll_exact(model_expod[1,2,idx], Y_tmp[idx,1])
    @test cube_tmp[1,3,idx] == LRMoE.expert_ll_exact(model_expod[1,3,idx], Y_tmp[idx,1])
    @test cube_tmp[1,4,idx] == LRMoE.expert_ll_exact(model_expod[1,4,idx], Y_tmp[idx,1])
    @test cube_tmp[2,1,idx] == LRMoE.expert_ll_exact(model_expod[2,1,idx], Y_tmp[idx,2])
    @test cube_tmp[2,2,idx] == LRMoE.expert_ll_exact(model_expod[2,2,idx], Y_tmp[idx,2])
    @test cube_tmp[2,3,idx] == LRMoE.expert_ll_exact(model_expod[2,3,idx], Y_tmp[idx,2])
    @test cube_tmp[2,4,idx] == LRMoE.expert_ll_exact(model_expod[2,4,idx], Y_tmp[idx,2])


    # ll related
    gate_tmp = rand(Distributions.Uniform(-1, 1), 15, 4)

    dim_agg = LRMoE.loglik_aggre_dim(cube_tmp)
    @test size(dim_agg) == (15, 4)
    @test size(LRMoE.loglik_aggre_gate_dim(gate_tmp, dim_agg)) == (15, 4)
    

end