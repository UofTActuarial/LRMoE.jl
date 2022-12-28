@testset "zibinomial pdf, cdf, etc." begin
    p0p0 = [0.01, 0.05, 0.10, 0.50, 0.90, 0.95, 0.99]
    nn = [1, 2, 5, 10, 25, 50, 100]
    pp = [0.01, 0.05, 0.10, 0.50, 0.90, 0.95, 0.99]
    x = collect(0.0:1:50.0)

    for p0 in p0p0, n in nn, p in pp
        l = Distributions.Binomial(n, p)
        r = LRMoE.ZIBinomialExpert(p0, n, p)
        @test LRMoE.logpdf.(r, x) ≈ Distributions.logpdf.(l, x)
        @test LRMoE.logcdf.(r, x) ≈ Distributions.logcdf.(l, x)
        @test LRMoE.pdf.(r, x) ≈ Distributions.pdf.(l, x)
        @test LRMoE.cdf.(r, x) ≈ Distributions.cdf.(l, x)
    end
end

@testset "zibinomial expert_ll" begin
    p0p0 = [0.01, 0.05, 0.10, 0.50, 0.90, 0.95, 0.99]
    nn = [1, 2, 5, 10, 25, 50, 100]
    pp = [0.01, 0.05, 0.10, 0.50, 0.90, 0.95, 0.99]
    x = collect(0.0:1:50.0)

    for p0 in p0p0, n in nn, p in pp
        tmp = LRMoE.ZIBinomialExpert(p0, n, p)

        # exact observations
        true_val = log.(1 - p0) .+ LRMoE.logpdf.(tmp, x)
        true_val[x .== 0] .= log(p0 + (1 - p0) .* exp.(LRMoE.logpdf.(tmp, 0.0)))
        @test isapprox(LRMoE.expert_ll_exact.(tmp, x), true_val)

        true_val = log.(1 - p0) .+ LRMoE.logpdf.(tmp, x)
        true_val[x .== 0] .= log(p0 + (1 - p0) .* exp.(LRMoE.logpdf.(tmp, 0.0)))
        @test isapprox(LRMoE.expert_ll.(tmp, 0.0, x, x, Inf), true_val)

        # right censoring
        true_val =
            log.(1 - p0) .+ logcdf.(tmp, Inf) .+
            log1mexp.(logcdf.(tmp, ceil.(x) .- 1) .- logcdf.(tmp, Inf))
        true_val[x .== 0] .= log(
            p0 +
            (1 - p0) .*
            exp.(
                logcdf.(tmp, Inf) .+
                log1mexp.(logcdf.(tmp, ceil.(0.0) .- 1) .- logcdf.(tmp, Inf))
            ),
        )
        @test isapprox(LRMoE.expert_ll.(tmp, 0.0, x, Inf, Inf), true_val, atol=1e-6)

        true_val =
            log.(1 - p0) .+ logcdf.(tmp, 1.25 .* x) .+
            log1mexp.(logcdf.(tmp, ceil.(0.75 .* x) .- 1) .- logcdf.(tmp, 1.25 .* x))
        true_val[x .== 0] .= log(
            p0 +
            (1 - p0) .*
            exp.(
                logcdf.(tmp, 1.25 .* 0.0) .+
                log1mexp.(
                    logcdf.(tmp, ceil.(0.75 .* 0.0) .- 1) .- logcdf.(tmp, 1.25 .* 0.0)
                )
            ),
        )
        @test isapprox(
            LRMoE.expert_ll.(tmp, 0.0, 0.75 .* x, 1.25 .* x, Inf), true_val, atol=1e-6
        )

        # exact observations
        @test isapprox(LRMoE.expert_tn.(tmp, 0.0, x, x, Inf), fill(0.0, length(x)))

        # censoring + truncation
        true_val =
            log.(1 - p0) .+ logcdf.(tmp, 1.5 .* x) .+
            log1mexp.(logcdf.(tmp, ceil.(0.25 .* x) .- 1) .- logcdf.(tmp, 1.5 .* x))
        true_val[x .== 0] .= log(
            p0 +
            (1 - p0) .*
            exp.(
                logcdf.(tmp, 1.5 .* 0.0) .+
                log1mexp.(
                    logcdf.(tmp, ceil.(0.25 .* 0.0) .- 1) .- logcdf.(tmp, 1.5 .* 0.0)
                )
            ),
        )
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
                log.(1 - p0) .+ logcdf.(tmp, 1.5 .* x) .+
                log1mexp.(logcdf.(tmp, ceil.(0.25 .* x) .- 1) .- logcdf.(tmp, 1.5 .* x))
            )
        true_val[x .== 0] .= log1mexp(
            log(p0 + (1 - p0) .* exp.(LRMoE.logpdf.(tmp, 0.0)))
        )
        @test isapprox(
            LRMoE.expert_tn_bar.(tmp, 0.25 .* x, 0.75 .* x, 1.25 .* x, 1.5 .* x),
            true_val,
            atol=1e-6,
        )
    end
end

@testset "zibinomial exposurize" begin
    d = ZIBinomialExpert()
    exposure_vec = [0.5, 1.0, 2.0, 5.0, 10.0]

    for e in exposure_vec
        @test LRMoE.exposurize_expert(d; exposure=e) == d
    end
end
