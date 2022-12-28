@testset "exposurize experts" begin
    param0_list = [0.1, 0.5, 0.9]
    param1_list = [10.0, 15.0, 20.0]
    param2_list = [0.1, 0.5, 0.9]
    param3_list = [5.0, 7.0, 10.0]

    sample_size = 21

    expo_list = [0.5, 1.0, 2.0, 5.0, 10.0]

    for param0 in param0_list,
        param1 in param1_list,
        param2 in param2_list,
        param3 in param3_list,
        expo in expo_list

        tmp = LRMoE.BinomialExpert(20, param0)
        @test tmp == LRMoE.exposurize_expert(tmp; exposure=expo)
        tmp = LRMoE.ZIBinomialExpert(0.20, 20, param0)
        @test tmp == LRMoE.exposurize_expert(tmp; exposure=expo)

        tmp = LRMoE.GammaCountExpert(param1, param2)
        if expo != 1.0
            @test tmp != LRMoE.exposurize_expert(tmp; exposure=expo)
        else
            @test tmp == LRMoE.exposurize_expert(tmp; exposure=expo)
        end
        @test isapprox(
            [LRMoE.params(tmp)...] .* [1, 1 / expo],
            [LRMoE.params(LRMoE.exposurize_expert(tmp; exposure=expo))...],
            atol=1e-6,
        )

        tmp = LRMoE.ZIGammaCountExpert(param0, param1, param2)
        if expo != 1.0
            @test tmp != LRMoE.exposurize_expert(tmp; exposure=expo)
        else
            @test tmp == LRMoE.exposurize_expert(tmp; exposure=expo)
        end
        @test isapprox(
            [LRMoE.params(tmp)...] .* [1, 1, 1 / expo],
            [LRMoE.params(LRMoE.exposurize_expert(tmp; exposure=expo))...],
            atol=1e-6,
        )

        tmp = LRMoE.NegativeBinomialExpert(param1, param0)
        if expo != 1.0
            @test tmp != LRMoE.exposurize_expert(tmp; exposure=expo)
        else
            @test tmp == LRMoE.exposurize_expert(tmp; exposure=expo)
        end
        @test [LRMoE.params(tmp)...] .* [expo, 1] ==
            [LRMoE.params(LRMoE.exposurize_expert(tmp; exposure=expo))...]

        tmp = LRMoE.ZINegativeBinomialExpert(param0, param1, param0)
        if expo != 1.0
            @test tmp != LRMoE.exposurize_expert(tmp; exposure=expo)
        else
            @test tmp == LRMoE.exposurize_expert(tmp; exposure=expo)
        end
        @test [LRMoE.params(tmp)...] .* [1, expo, 1] ==
            [LRMoE.params(LRMoE.exposurize_expert(tmp; exposure=expo))...]

        tmp = LRMoE.PoissonExpert(param1)
        if expo != 1.0
            @test tmp != LRMoE.exposurize_expert(tmp; exposure=expo)
        else
            @test tmp == LRMoE.exposurize_expert(tmp; exposure=expo)
        end
        @test [LRMoE.params(tmp)...] .* [expo] ==
            [LRMoE.params(LRMoE.exposurize_expert(tmp; exposure=expo))...]
        if expo != 1.0
            @test tmp != LRMoE.exposurize_expert(tmp; exposure=expo)
        else
            @test tmp == LRMoE.exposurize_expert(tmp; exposure=expo)
        end
        @test [LRMoE.params(tmp)...] .* [expo] ==
            [LRMoE.params(LRMoE.exposurize_expert(tmp; exposure=expo))...]

        model = [
            LogNormalExpert(param1, param2) ZIGammaExpert(param0, param1, param2) WeibullExpert(param1, param2) BurrExpert(param1, param2, param3)
            BinomialExpert(20, param0) PoissonExpert(param1) ZIGammaCountExpert(param0, param1, param2) ZINegativeBinomialExpert(param0, param1, param0)
        ]

        expos = fill(expo, sample_size)

        @test size(LRMoE.exposurize_model(model; exposure=expos)) == (2, 4, sample_size)

        @test mean.(LRMoE.exposurize_model(model; exposure=expos))[1, 1, :] ==
            fill(mean(model[1, 1]), sample_size)
        @test mean.(LRMoE.exposurize_model(model; exposure=expos))[1, 2, :] ==
            fill(mean(model[1, 2]), sample_size)
        @test mean.(LRMoE.exposurize_model(model; exposure=expos))[1, 3, :] ==
            fill(mean(model[1, 3]), sample_size)
        @test mean.(LRMoE.exposurize_model(model; exposure=expos))[1, 4, :] ==
            fill(mean(model[1, 4]), sample_size)

        @test mean.(LRMoE.exposurize_model(model; exposure=expos))[2, 1, :] ==
            fill(mean(model[2, 1]), sample_size)
        @test mean.(LRMoE.exposurize_model(model; exposure=expos))[2, 2, :] ==
            fill(mean(model[2, 2]) .* expo, sample_size)

        model_expod = LRMoE.exposurize_model(model; exposure=expos)
        Y_tmp = hcat(
            collect(0.0:0.5:10), collect(0:1:20)
        )

        @test size(LRMoE.expert_ll_ind_mat_exact(Y_tmp[1, :], model_expod[:, :, 1])) ==
            (2, 4)
        @test size(LRMoE.expert_ll_ind_mat_exact(Y_tmp[5, :], model_expod[:, :, 5])) ==
            (2, 4)
        @test size(LRMoE.expert_ll_ind_mat_exact(Y_tmp[12, :], model_expod[:, :, 12])) ==
            (2, 4)
        @test size(LRMoE.expert_ll_list_exact(Y_tmp, model_expod)) == (2, 4, sample_size)

        for idx in [1, 5, 10, 20, 21]
            mat_tmp = LRMoE.expert_ll_ind_mat_exact(Y_tmp[idx, :], model_expod[:, :, idx])
            @test mat_tmp[1, 1] ==
                LRMoE.expert_ll_exact(model_expod[1, 1, idx], Y_tmp[idx, 1])
            @test mat_tmp[1, 2] ==
                LRMoE.expert_ll_exact(model_expod[1, 2, idx], Y_tmp[idx, 1])
            @test mat_tmp[1, 3] ==
                LRMoE.expert_ll_exact(model_expod[1, 3, idx], Y_tmp[idx, 1])
            @test mat_tmp[1, 4] ==
                LRMoE.expert_ll_exact(model_expod[1, 4, idx], Y_tmp[idx, 1])
            @test mat_tmp[2, 1] ==
                LRMoE.expert_ll_exact(model_expod[2, 1, idx], Y_tmp[idx, 2])
            @test mat_tmp[2, 2] ==
                LRMoE.expert_ll_exact(model_expod[2, 2, idx], Y_tmp[idx, 2])
            @test mat_tmp[2, 3] ==
                LRMoE.expert_ll_exact(model_expod[2, 3, idx], Y_tmp[idx, 2])
            @test mat_tmp[2, 3] ==
                LRMoE.expert_ll_exact(model_expod[2, 3, idx], Y_tmp[idx, 2])
        end

        for idx in [1, 5, 10, 20, 21]
            cube_tmp = LRMoE.expert_ll_list_exact(Y_tmp, model_expod)
            @test cube_tmp[1, 1, idx] ==
                LRMoE.expert_ll_exact(model_expod[1, 1, idx], Y_tmp[idx, 1])
            @test cube_tmp[1, 2, idx] ==
                LRMoE.expert_ll_exact(model_expod[1, 2, idx], Y_tmp[idx, 1])
            @test cube_tmp[1, 3, idx] ==
                LRMoE.expert_ll_exact(model_expod[1, 3, idx], Y_tmp[idx, 1])
            @test cube_tmp[1, 4, idx] ==
                LRMoE.expert_ll_exact(model_expod[1, 4, idx], Y_tmp[idx, 1])
            @test cube_tmp[2, 1, idx] ==
                LRMoE.expert_ll_exact(model_expod[2, 1, idx], Y_tmp[idx, 2])
            @test cube_tmp[2, 2, idx] ==
                LRMoE.expert_ll_exact(model_expod[2, 2, idx], Y_tmp[idx, 2])
            @test cube_tmp[2, 3, idx] ==
                LRMoE.expert_ll_exact(model_expod[2, 3, idx], Y_tmp[idx, 2])
            @test cube_tmp[2, 4, idx] ==
                LRMoE.expert_ll_exact(model_expod[2, 4, idx], Y_tmp[idx, 2])
        end
    end
end

@testset "exposurize model" begin
    param0_list = [0.1, 0.5, 0.9]
    param1_list = [10.0, 15.0, 20.0]
    param2_list = [0.1, 0.5, 0.9]
    param3_list = [5.0, 7.0, 10.0]

    sample_size = 21

    expo_list = [0.5, 1.0, 2.0, 5.0, 10.0]

    for param0 in param0_list,
        param1 in param1_list,
        param2 in param2_list,
        param3 in param3_list,
        expo in expo_list

        model = [
            LogNormalExpert(param1, param2) ZIGammaExpert(param0, param1, param2) WeibullExpert(param1, param2) BurrExpert(param1, param2, param3)
            BinomialExpert(20, param0) PoissonExpert(param1) ZIGammaCountExpert(param0, param1, param2) ZINegativeBinomialExpert(param0, param1, param0)
        ]

        expos = fill(expo, sample_size)

        @test size(LRMoE.exposurize_model(model; exposure=expos)) == (2, 4, sample_size)

        @test mean.(LRMoE.exposurize_model(model; exposure=expos))[1, 1, :] ==
            fill(mean(model[1, 1]), sample_size)
        @test mean.(LRMoE.exposurize_model(model; exposure=expos))[1, 2, :] ==
            fill(mean(model[1, 2]), sample_size)
        @test mean.(LRMoE.exposurize_model(model; exposure=expos))[1, 3, :] ==
            fill(mean(model[1, 3]), sample_size)
        @test mean.(LRMoE.exposurize_model(model; exposure=expos))[1, 4, :] ==
            fill(mean(model[1, 4]), sample_size)

        @test mean.(LRMoE.exposurize_model(model; exposure=expos))[2, 1, :] ==
            fill(mean(model[2, 1]), sample_size)
        @test mean.(LRMoE.exposurize_model(model; exposure=expos))[2, 2, :] ==
            fill(mean(model[2, 2]) .* expo, sample_size)

        model_expod = LRMoE.exposurize_model(model; exposure=expos)
        Y_tmp = hcat(
            collect(0.0:0.5:10), collect(0:1:20)
        )

        @test size(LRMoE.expert_ll_ind_mat_exact(Y_tmp[1, :], model_expod[:, :, 1])) ==
            (2, 4)
        @test size(LRMoE.expert_ll_ind_mat_exact(Y_tmp[5, :], model_expod[:, :, 5])) ==
            (2, 4)
        @test size(LRMoE.expert_ll_ind_mat_exact(Y_tmp[12, :], model_expod[:, :, 12])) ==
            (2, 4)
        @test size(LRMoE.expert_ll_list_exact(Y_tmp, model_expod)) == (2, 4, sample_size)

        for idx in [1, 5, 10, 20, 21]
            mat_tmp = LRMoE.expert_ll_ind_mat_exact(Y_tmp[idx, :], model_expod[:, :, idx])
            @test mat_tmp[1, 1] ==
                LRMoE.expert_ll_exact(model_expod[1, 1, idx], Y_tmp[idx, 1])
            @test mat_tmp[1, 2] ==
                LRMoE.expert_ll_exact(model_expod[1, 2, idx], Y_tmp[idx, 1])
            @test mat_tmp[1, 3] ==
                LRMoE.expert_ll_exact(model_expod[1, 3, idx], Y_tmp[idx, 1])
            @test mat_tmp[1, 4] ==
                LRMoE.expert_ll_exact(model_expod[1, 4, idx], Y_tmp[idx, 1])
            @test mat_tmp[2, 1] ==
                LRMoE.expert_ll_exact(model_expod[2, 1, idx], Y_tmp[idx, 2])
            @test mat_tmp[2, 2] ==
                LRMoE.expert_ll_exact(model_expod[2, 2, idx], Y_tmp[idx, 2])
            @test mat_tmp[2, 3] ==
                LRMoE.expert_ll_exact(model_expod[2, 3, idx], Y_tmp[idx, 2])
            @test mat_tmp[2, 3] ==
                LRMoE.expert_ll_exact(model_expod[2, 3, idx], Y_tmp[idx, 2])
        end

        for idx in [1, 5, 10, 20, 21]
            cube_tmp = LRMoE.expert_ll_list_exact(Y_tmp, model_expod)
            @test cube_tmp[1, 1, idx] ==
                LRMoE.expert_ll_exact(model_expod[1, 1, idx], Y_tmp[idx, 1])
            @test cube_tmp[1, 2, idx] ==
                LRMoE.expert_ll_exact(model_expod[1, 2, idx], Y_tmp[idx, 1])
            @test cube_tmp[1, 3, idx] ==
                LRMoE.expert_ll_exact(model_expod[1, 3, idx], Y_tmp[idx, 1])
            @test cube_tmp[1, 4, idx] ==
                LRMoE.expert_ll_exact(model_expod[1, 4, idx], Y_tmp[idx, 1])
            @test cube_tmp[2, 1, idx] ==
                LRMoE.expert_ll_exact(model_expod[2, 1, idx], Y_tmp[idx, 2])
            @test cube_tmp[2, 2, idx] ==
                LRMoE.expert_ll_exact(model_expod[2, 2, idx], Y_tmp[idx, 2])
            @test cube_tmp[2, 3, idx] ==
                LRMoE.expert_ll_exact(model_expod[2, 3, idx], Y_tmp[idx, 2])
            @test cube_tmp[2, 4, idx] ==
                LRMoE.expert_ll_exact(model_expod[2, 4, idx], Y_tmp[idx, 2])
        end

        Y_tmp = hcat(
            0.2 .* Y_tmp[:, 1],
            0.8 .* Y_tmp[:, 1],
            1.2 .* Y_tmp[:, 1],
            5 .* Y_tmp[:, 1],
            fill(0, length(Y_tmp[:, 2])),
            Y_tmp[:, 2],
            Y_tmp[:, 2],
            fill(Inf, length(Y_tmp[:, 2])),
        )

        @test size(LRMoE.expert_ll_ind_mat(Y_tmp[1, :], model_expod[:, :, 1])) == (2, 4)
        @test size(LRMoE.expert_ll_ind_mat(Y_tmp[5, :], model_expod[:, :, 5])) == (2, 4)
        @test size(LRMoE.expert_ll_ind_mat(Y_tmp[12, :], model_expod[:, :, 12])) == (2, 4)
        @test size(LRMoE.expert_ll_list(Y_tmp, model_expod)) == (2, 4, sample_size)

        for idx in [1, 5, 10, 20, 21]
            mat_tmp = LRMoE.expert_ll_ind_mat(Y_tmp[idx, :], model_expod[:, :, idx])
            @test mat_tmp[1, 1] == LRMoE.expert_ll(
                model_expod[1, 1, idx],
                Y_tmp[idx, 1],
                Y_tmp[idx, 2],
                Y_tmp[idx, 3],
                Y_tmp[idx, 4],
            )
            @test mat_tmp[1, 2] == LRMoE.expert_ll(
                model_expod[1, 2, idx],
                Y_tmp[idx, 1],
                Y_tmp[idx, 2],
                Y_tmp[idx, 3],
                Y_tmp[idx, 4],
            )
            @test mat_tmp[1, 3] == LRMoE.expert_ll(
                model_expod[1, 3, idx],
                Y_tmp[idx, 1],
                Y_tmp[idx, 2],
                Y_tmp[idx, 3],
                Y_tmp[idx, 4],
            )
            @test mat_tmp[1, 4] == LRMoE.expert_ll(
                model_expod[1, 4, idx],
                Y_tmp[idx, 1],
                Y_tmp[idx, 2],
                Y_tmp[idx, 3],
                Y_tmp[idx, 4],
            )
            @test mat_tmp[2, 1] == LRMoE.expert_ll(
                model_expod[2, 1, idx],
                Y_tmp[idx, 5],
                Y_tmp[idx, 6],
                Y_tmp[idx, 7],
                Y_tmp[idx, 8],
            )
            @test mat_tmp[2, 2] == LRMoE.expert_ll(
                model_expod[2, 2, idx],
                Y_tmp[idx, 5],
                Y_tmp[idx, 6],
                Y_tmp[idx, 7],
                Y_tmp[idx, 8],
            )
            @test mat_tmp[2, 3] == LRMoE.expert_ll(
                model_expod[2, 3, idx],
                Y_tmp[idx, 5],
                Y_tmp[idx, 6],
                Y_tmp[idx, 7],
                Y_tmp[idx, 8],
            )
            @test mat_tmp[2, 3] == LRMoE.expert_ll(
                model_expod[2, 3, idx],
                Y_tmp[idx, 5],
                Y_tmp[idx, 6],
                Y_tmp[idx, 7],
                Y_tmp[idx, 8],
            )
        end

        cube_tmp = LRMoE.expert_ll_list(Y_tmp, model_expod)

        for idx in [1, 5, 10, 20, 21]
            @test cube_tmp[1, 1, idx] == LRMoE.expert_ll(
                model_expod[1, 1, idx],
                Y_tmp[idx, 1],
                Y_tmp[idx, 2],
                Y_tmp[idx, 3],
                Y_tmp[idx, 4],
            )
            @test cube_tmp[1, 2, idx] == LRMoE.expert_ll(
                model_expod[1, 2, idx],
                Y_tmp[idx, 1],
                Y_tmp[idx, 2],
                Y_tmp[idx, 3],
                Y_tmp[idx, 4],
            )
            @test cube_tmp[1, 3, idx] == LRMoE.expert_ll(
                model_expod[1, 3, idx],
                Y_tmp[idx, 1],
                Y_tmp[idx, 2],
                Y_tmp[idx, 3],
                Y_tmp[idx, 4],
            )
            @test cube_tmp[1, 4, idx] == LRMoE.expert_ll(
                model_expod[1, 4, idx],
                Y_tmp[idx, 1],
                Y_tmp[idx, 2],
                Y_tmp[idx, 3],
                Y_tmp[idx, 4],
            )
            @test cube_tmp[2, 1, idx] == LRMoE.expert_ll(
                model_expod[2, 1, idx],
                Y_tmp[idx, 5],
                Y_tmp[idx, 6],
                Y_tmp[idx, 7],
                Y_tmp[idx, 8],
            )
            @test cube_tmp[2, 2, idx] == LRMoE.expert_ll(
                model_expod[2, 2, idx],
                Y_tmp[idx, 5],
                Y_tmp[idx, 6],
                Y_tmp[idx, 7],
                Y_tmp[idx, 8],
            )
            @test cube_tmp[2, 3, idx] == LRMoE.expert_ll(
                model_expod[2, 3, idx],
                Y_tmp[idx, 5],
                Y_tmp[idx, 6],
                Y_tmp[idx, 7],
                Y_tmp[idx, 8],
            )
            @test cube_tmp[2, 4, idx] == LRMoE.expert_ll(
                model_expod[2, 4, idx],
                Y_tmp[idx, 5],
                Y_tmp[idx, 6],
                Y_tmp[idx, 7],
                Y_tmp[idx, 8],
            )
        end

        # ll related
        Random.seed!(1234)

        gate_tmp = rand(Distributions.Uniform(-1, 1), sample_size, 4)

        dim_agg = LRMoE.loglik_aggre_dim(cube_tmp)
        @test size(dim_agg) == (sample_size, 4)
        @test size(LRMoE.loglik_aggre_gate_dim(gate_tmp, dim_agg)) == (sample_size, 4)

        # Simulation related
        # X_tmp = rand(Distributions.Uniform(-1, 1), 15, 7)
        # α_tmp = rand(Distributions.Uniform(-1, 1), 4, 7)

        # model = [LogNormalExpert(param1, param2) ZIGammaExpert(param0, param1, param2) WeibullExpert(param1, param2) BurrExpert(param1, param2, param3);
        #         BinomialExpert(20, param0) PoissonExpert(param1) ZIGammaCountExpert(param0, param1, param2) ZINegativeBinomialExpert(param0, param1, param0)]

        # expo = rand(Distributions.Uniform(0.5, 20), 15)

        # model_expod = LRMoE.exposurize_model(model, exposure = expo)

        # sim_tmp = LRMoE.sim_dataset(α_tmp, X_tmp, model, exposure = expo)
        # mean(sim_tmp, dims = 1)

        # sim_tmp = LRMoE.sim_dataset(α_tmp, X_tmp, model)
        # mean(sim_tmp, dims = 1)
    end
end