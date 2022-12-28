# Won't test: k <= 1.0 due to spurious likelihood at zero
@testset "weibull pdf, cdf, etc." begin
    kk = [1.5, 2.0, 5.0, 10.0]
    θθ = [1.0, 0.5, 0.1, 0.01]
    x = collect(0.0:0.5:100.0)

    for k in kk, θ in θθ
        l = Distributions.Weibull(k, θ)
        r = LRMoE.WeibullExpert(k, θ)
        @test LRMoE.logpdf.(r, x) ≈ Distributions.logpdf.(l, x)
        @test LRMoE.logcdf.(r, x) ≈ Distributions.logcdf.(l, x)
        @test LRMoE.pdf.(r, x) ≈ Distributions.pdf.(l, x)
        @test LRMoE.cdf.(r, x) ≈ Distributions.cdf.(l, x)
    end
end

@testset "weibull expert_ll" begin
    kk = [1.5, 2.0, 5.0, 10.0]
    θθ = [1.0, 0.5, 0.1, 0.01]
    x = collect(0.0:0.5:100.0)

    for k in kk, θ in θθ
        tmp = LRMoE.WeibullExpert(k, θ)

        # exact observations
        @test isapprox(LRMoE.expert_ll_exact.(tmp, x), LRMoE.logpdf.(tmp, x))
        @test isapprox(LRMoE.expert_ll.(tmp, 0.0, x, x, Inf), LRMoE.logpdf.(tmp, x))

        # right censoring
        true_val = logcdf.(tmp, Inf) .+ log1mexp.(logcdf.(tmp, x) .- logcdf.(tmp, Inf))
        @test isapprox(LRMoE.expert_ll.(tmp, 0.0, x, Inf, Inf), true_val, atol=1e-6)

        true_val =
            logcdf.(tmp, 1.25 .* x) .+
            log1mexp.(logcdf.(tmp, 0.75 .* x) .- logcdf.(tmp, 1.25 .* x))
        true_val[x .== 0.0] .= -Inf
        @test isapprox(
            LRMoE.expert_ll.(tmp, 0.0, 0.75 .* x, 1.25 .* x, Inf), true_val, atol=1e-6
        )

        # exact observations
        @test isapprox(LRMoE.expert_tn.(tmp, 0.0, x, x, Inf), fill(0.0, length(x)))

        # censoring + truncation
        true_val =
            logcdf.(tmp, 1.5 .* x) .+
            log1mexp.(logcdf.(tmp, 0.25 .* x) .- logcdf.(tmp, 1.5 .* x))
        true_val[x .== 0.0] .= -Inf
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
        true_val[x .== 0.0] .= 0.0
        @test isapprox(
            LRMoE.expert_tn_bar.(tmp, 0.25 .* x, 0.75 .* x, 1.25 .* x, 1.5 .* x),
            true_val,
            atol=1e-6,
        )
    end
end

@testset "weibull exposurize" begin
    d = WeibullExpert()
    exposure_vec = [0.5, 1.0, 2.0, 5.0, 10.0]

    for e in exposure_vec
        @test LRMoE.exposurize_expert(d; exposure=e) == d
    end
end

@testset "weibull lev/excess" begin
    kk = [1.5, 2.0, 5.0, 10.0]
    θθ = [1.0, 0.5, 0.1, 0.01]
    x = [0.0, Inf]

    for k in kk, θ in θθ
        tmp = WeibullExpert(k, θ)
        @test LRMoE.lev(tmp, 0.0) == 0.0
        @test LRMoE.excess(tmp, 0.0) == mean(Weibull(k, θ))
        @test LRMoE.lev(tmp, Inf) == mean(Weibull(k, θ))
        @test LRMoE.excess(tmp, Inf) == 0.0
    end
end