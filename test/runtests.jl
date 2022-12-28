using LRMoE
using Distributions
using QuadGK
using Random
using StatsFuns
using Test

const tests = [
    # "dummytest",
    # "pdfcdf",
    # "expert_loglik",
    # "loglik",
    # "fit_functions"

    "gating",

    "experts/continuous/burr",
    "experts/continuous/gamma",
    "experts/continuous/inversegaussian",
    "experts/continuous/lognormal",
    "experts/continuous/weibull",

    "experts/continuous/ziburr",
    "experts/continuous/zigamma",
    "experts/continuous/ziinversegaussian",
    "experts/continuous/zilognormal",
    "experts/continuous/ziweibull",

    "experts/discrete/poisson",

    "experts/discrete/zipoisson",

]

printstyled("Running tests:\n", color=:blue)

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