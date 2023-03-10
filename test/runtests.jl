using LRMoE
using Distributions
using QuadGK
using Random
using StatsFuns
using Test

const tests = [
    # gating
    "gating",
    # exposurize
    "experts/ll/exposurize",
    # continuous experts
    "experts/continuous/burr",
    "experts/continuous/gamma",
    "experts/continuous/inversegaussian",
    "experts/continuous/lognormal",
    "experts/continuous/weibull",
    # zi continuous experts
    "experts/continuous/ziburr",
    "experts/continuous/zigamma",
    "experts/continuous/ziinversegaussian",
    "experts/continuous/zilognormal",
    "experts/continuous/ziweibull",
    # discrete experts
    "experts/discrete/binomial",
    "experts/discrete/gammacount",
    "experts/discrete/negativebinomial",
    "experts/discrete/poisson",
    # zi discrete experts
    "experts/discrete/zibinomial",
    "experts/discrete/zigammacount",
    "experts/discrete/zinegativebinomial",
    "experts/discrete/zipoisson",
    # loglik
    # "loglik",
]

printstyled("Running tests:\n"; color=:blue)

for t in tests
    @testset "Test $t" begin
        Random.seed!(345679)
        include("$t.jl")
    end
end

# print method ambiguities
println("Potentially stale exports: ")
display(Test.detect_ambiguities(LRMoE))
println()