@testset "zi_burr pdf, cdf, etc." begin
    pp = [0.05, 0.10, 0.50, 0.90]
    kk = [1.5, 2.0, 5.0, 10.0]
    cc = [1.5, 2.0, 5.0, 10.0]
    λλ = [1.0, 0.5, 0.1, 0.01]
    x = collect(0.0:0.5:100.0)

    for p in pp, k in kk, c in cc, λ in λλ
        l = LRMoE.Burr(k, c, λ)
        r = LRMoE.ZIBurrExpert(p, k, c, λ)
        @test LRMoE.logpdf.(r, x) ≈ Distributions.logpdf.(l, x)
        @test LRMoE.logcdf.(r, x) ≈ Distributions.logcdf.(l, x)
        @test LRMoE.pdf.(r, x) ≈ Distributions.pdf.(l, x)
        @test LRMoE.cdf.(r, x) ≈ Distributions.cdf.(l, x)
    end
end

@testset "zi_burr expert_ll" begin
    pp = [0.05, 0.10, 0.50, 0.90]
    kk = [1.5, 2.0, 5.0, 10.0]
    cc = [1.5, 2.0, 5.0, 10.0]
    λλ = [1.0, 0.5, 0.1, 0.01]
    x = collect(0.0:0.5:100.0)

    for p in pp, k in kk, c in cc, λ in λλ
        tmp = LRMoE.ZIBurrExpert(p, k, c, λ)

        # exact observations
        true_val = log(1 - p) .+ LRMoE.logpdf.(tmp, x)
        true_val[x .== 0.0] .= log(p)
        @test isapprox(LRMoE.expert_ll_exact.(tmp, x), true_val, atol=1e-6)

        # right censoring
        true_val =
            log.(
                (1 - p) .*
                exp.(logcdf.(tmp, Inf) .+ log1mexp.(logcdf.(tmp, x) .- logcdf.(tmp, Inf)))
            )
        true_val[x .== 0.0] .=
            log.(
                p .+
                (1 - p) .*
                exp.(
                    logcdf.(tmp, Inf) .+ log1mexp.(logcdf.(tmp, 0.0) .- logcdf.(tmp, Inf))
                )
            )
        @test isapprox(LRMoE.expert_ll.(tmp, 0.0, x, Inf, Inf), true_val, atol=1e-6)

        true_val =
            log.(
                (1 - p) .*
                exp.(
                    logcdf.(tmp, 1.25 .* x) .+
                    log1mexp.(logcdf.(tmp, 0.75 .* x) .- logcdf.(tmp, 1.25 .* x))
                )
            )
        true_val[x .== 0.0] .= log(p)
        @test isapprox(
            LRMoE.expert_ll.(tmp, 0.0, 0.75 .* x, 1.25 .* x, Inf), true_val, atol=1e-6
        )

        # exact observations
        true_val = fill(0.0, length(x))
        @test isapprox(LRMoE.expert_tn.(tmp, 0.0, x, x, Inf), true_val, atol=1e-6)

        # censoring + truncation
        true_val =
            log.(
                (1 - p) .*
                exp.(
                    logcdf.(tmp, 1.5 .* x) .+
                    log1mexp.(logcdf.(tmp, 0.25 .* x) .- logcdf.(tmp, 1.5 .* x))
                )
            )
        true_val[x .== 0.0] .= log(p)
        @test isapprox(
            LRMoE.expert_tn.(tmp, 0.25 .* x, 0.75 .* x, 1.25 .* x, 1.5 .* x),
            true_val,
            atol=1e-6,
        )

        # exact observations
        true_val = fill(-Inf, length(x))
        @test isapprox(LRMoE.expert_tn_bar.(tmp, 0.0, x, x, Inf), true_val, atol=1e-6)

        # censoring + truncation
        true_val =
            log.(
                p .+
                (1 - p) .*
                exp.(
                    log1mexp.(
                        logcdf.(tmp, 1.5 .* x) .+
                        log1mexp.(logcdf.(tmp, 0.25 .* x) .- logcdf.(tmp, 1.5 .* x))
                    )
                )
            )
        true_val[x .== 0.0] .= log(1 - p)
        @test isapprox(
            LRMoE.expert_tn_bar.(tmp, 0.25 .* x, 0.75 .* x, 1.25 .* x, 1.5 .* x),
            true_val,
            atol=1e-6,
        )
    end
end

@testset "zi_burr exposurize" begin
    d = ZIBurrExpert()
    exposure_vec = [0.5, 1.0, 2.0, 5.0, 10.0]

    for e in exposure_vec
        @test LRMoE.exposurize_expert(d; exposure=e) == d
    end
end

@testset "zi_burr lev/excess" begin
    pp = [0.05, 0.10, 0.50, 0.90]
    kk = [1.5, 2.0, 5.0, 10.0]
    cc = [1.5, 2.0, 5.0, 10.0]
    λλ = [1.0, 0.5, 0.1, 0.01]
    x = [0.0, Inf]

    for p in pp, k in kk, c in cc, λ in λλ
        tmp = ZIBurrExpert(p, k, c, λ)
        @test LRMoE.lev(tmp, 0.0) == 0.0
        @test LRMoE.excess(tmp, 0.0) == (1 - p) * mean(Burr(k, c, λ))
        @test LRMoE.lev(tmp, Inf) == (1 - p) * mean(Burr(k, c, λ))
        @test LRMoE.excess(tmp, Inf) == 0.0
    end
end
