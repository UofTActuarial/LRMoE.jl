@testset "lognormal pdf, cdf, etc." begin
    μμ = [1.0, 2.0, 5.0, 10.0]
    σσ = [1.0, 0.5, 0.1, 0.01]
    x = collect(0.0:0.5:100.0)

    for μ in μμ, σ in σσ
        l = Distributions.LogNormal(μ, σ)
        r = LRMoE.LogNormalExpert(μ, σ)
        @test LRMoE.logpdf(r, x) ≈ Distributions.logpdf.(l, x)
        @test LRMoE.logcdf(r, x) ≈ Distributions.logcdf.(l, x)
        @test LRMoE.pdf(r, x) ≈ Distributions.pdf.(l, x)
        @test LRMoE.cdf(r, x) ≈ Distributions.cdf.(l, x)
    end
end

@testset "lognormal expert_ll" begin
    μμ = [1.0, 2.0, 5.0, 10.0]
    σσ = [1.0, 0.5, 0.1, 0.01]
    x = collect(0.0:0.5:100.0)

    for μ in μμ, σ in σσ
        tmp = LRMoE.LogNormalExpert(μ, σ)

        # exact observations
        @test isapprox(LRMoE.expert_ll_exact.(tmp, x), LRMoE.logpdf.(tmp, x))
        @test isapprox(LRMoE.expert_ll.(tmp, 0.0, x, x, Inf), LRMoE.logpdf.(tmp, x))

        # right censoring
        true_val = logcdf.(tmp, Inf) .+ log1mexp.(logcdf.(tmp, x) .- logcdf.(tmp, Inf))
        @test isapprox(LRMoE.expert_ll.(tmp, 0.0, x, Inf, Inf), true_val, atol=1e-6)

        true_val =
            logcdf.(tmp, 1.25 .* x) .+
            log1mexp.(logcdf.(tmp, 0.75 .* x) .- logcdf.(tmp, 1.25 .* x))
        true_val[isnan.(true_val)] .= -Inf # x == 0.0
        @test isapprox(
            LRMoE.expert_ll.(tmp, 0.0, 0.75 .* x, 1.25 .* x, Inf), true_val, atol=1e-6
        )

        # exact observations
        @test isapprox(LRMoE.expert_tn.(tmp, 0.0, x, x, Inf), fill(0.0, length(x)))

        # censoring + truncation
        true_val =
            logcdf.(tmp, 1.5 .* x) .+
            log1mexp.(logcdf.(tmp, 0.25 .* x) .- logcdf.(tmp, 1.5 .* x))
        true_val[isnan.(true_val)] .= -Inf # x == 0.0
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
                log1mexp.(logcdf.(tmp, 0.25 .* x) .- logcdf.(tmp, 1.5 .* x))
            )
        true_val[isnan.(true_val)] .= 0.0 # x == 0.0
        @test isapprox(
            LRMoE.expert_tn_bar.(tmp, 0.25 .* x, 0.75 .* x, 1.25 .* x, 1.5 .* x),
            true_val,
            atol=1e-6,
        )
    end
end

@testset "lognormal exposurize" begin
    d = LogNormalExpert()
    exposure_vec = [0.5, 1.0, 2.0, 5.0, 10.0]

    for e in exposure_vec
        @test LRMoE.exposurize_expert(d; exposure=e) == d
    end
end

@testset "lognormal lev/excess" begin
    μμ = [1.0, 2.0, 5.0, 10.0]
    σσ = [1.0, 0.5, 0.1, 0.01]
    x = [0.0, Inf]

    for μ in μμ, σ in σσ
        tmp = LogNormalExpert(μ, σ)
        @test LRMoE.lev(tmp, 0.0) == 0.0
        @test LRMoE.excess(tmp, 0.0) == mean(LogNormal(μ, σ))
        @test LRMoE.lev(tmp, Inf) == mean(LogNormal(μ, σ))
        @test LRMoE.excess(tmp, Inf) == 0.0
    end
end