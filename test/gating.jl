@testset "logit gating" begin
    Random.seed!(1234)

    d = Distributions.Uniform(-2.0, 2.0)

    for i in 1:5
        α = rand(d, 5, 20)
        x = rand(d, 100, 20)
        gating = LogitGating(α, x)
        ax = x * α'
        result = ax .- log.(sum(exp.(ax); dims=2))
        @test gating ≈ result
        @test sum(exp.(gating); dims=2) ≈ fill(1, size(gating)[1])
    end
end