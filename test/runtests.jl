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

    "experts/continuous/gamma",
    "experts/continuous/lognormal",

    "experts/continuous/zilognormal",

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