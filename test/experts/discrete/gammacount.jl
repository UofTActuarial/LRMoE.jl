@testset "gammacount pdf, cdf, etc." begin
    mm = [1.5, 2, 5, 10, 25]
    ss = [1.0, 2.0, 5.0, 10.0, 25.0]
    x = collect(0.0:1:50.0)

    for m in mm, s in ss
        l = LRMoE.GammaCount(m, s)
        r = LRMoE.GammaCountExpert(m, s)
        @test LRMoE.logpdf.(r, x) ≈ Distributions.logpdf.(l, x)
        @test LRMoE.logcdf.(r, x) ≈ Distributions.logcdf.(l, x)
        @test LRMoE.pdf.(r, x) ≈ Distributions.pdf.(l, x)
        @test LRMoE.cdf.(r, x) ≈ Distributions.cdf.(l, x)
    end
end

@testset "gammacount expert_ll" begin
    mm = [1.5, 2, 5, 10, 25]
    ss = [1.0, 2.0, 5.0, 10.0, 25.0]
    x = collect(0.0:1:50.0)

    for m in mm, s in ss
        tmp = LRMoE.GammaCountExpert(m, s)

        # exact observations
        @test isapprox(LRMoE.expert_ll_exact.(tmp, x), LRMoE.logpdf.(tmp, x))
        @test isapprox(LRMoE.expert_ll.(tmp, 0.0, x, x, Inf), LRMoE.logpdf.(tmp, x))

        # right censoring
        true_val =
            logcdf.(tmp, Inf) .+ log1mexp.(logcdf.(tmp, ceil.(x) .- 1) .- logcdf.(tmp, Inf))
        @test isapprox(LRMoE.expert_ll.(tmp, 0.0, x, Inf, Inf), true_val, atol=1e-6)

        true_val =
            logcdf.(tmp, 1.25 .* x) .+
            log1mexp.(logcdf.(tmp, ceil.(0.75 .* x) .- 1) .- logcdf.(tmp, 1.25 .* x))
        @test isapprox(
            LRMoE.expert_ll.(tmp, 0.0, 0.75 .* x, 1.25 .* x, Inf), true_val, atol=1e-6
        )

        # exact observations
        @test isapprox(LRMoE.expert_tn.(tmp, 0.0, x, x, Inf), fill(0.0, length(x)))

        # censoring + truncation
        true_val =
            logcdf.(tmp, 1.5 .* x) .+
            log1mexp.(logcdf.(tmp, ceil.(0.25 .* x) .- 1) .- logcdf.(tmp, 1.5 .* x))
        @test isapprox(
            LRMoE.expert_tn.(tmp, 0.25 .* x, 0.75 .* x, 1.25 .* x, 1.5 .* x),
            true_val,
            atol=1e-6,
        )

        # exact observations
        @test isapprox(LRMoE.expert_tn_bar.(tmp, 0.0, x, x, Inf), fill(-Inf, length(x)))

        # censoring + truncation
        true_val =
            log1mexp.(
                logcdf.(tmp, 1.5 .* x) .+
                log1mexp.(logcdf.(tmp, ceil.(0.25 .* x) .- 1) .- logcdf.(tmp, 1.5 .* x))
            )
        @test isapprox(
            LRMoE.expert_tn_bar.(tmp, 0.25 .* x, 0.75 .* x, 1.25 .* x, 1.5 .* x),
            true_val,
            atol=1e-6,
        )
    end
end

# @testset "gammacount exposurize" begin
#     d = GammaCountExpert()
#     exposure_vec = [0.5, 1.0, 2.0, 5.0, 10.0]

#     for e in exposure_vec
#         @test mean(LRMoE.exposurize_expert(d; exposure=e)) == mean(d) * e
#     end
# end
