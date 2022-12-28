using Test
using Distributions
using StatsFuns

using LRMoE

@testset "count with exposures" begin

    # mean(LRMoE.GammaCountExpert(1.0, 1.0))
    # mean(LRMoE.GammaCountExpert(2.0, 1.0))
    # mean(LRMoE.GammaCountExpert(3.0, 1.0))
    # mean(LRMoE.GammaCountExpert(4.0, 1.0))
    # mean(LRMoE.GammaCountExpert(5.0, 1.0))

    # mean(LRMoE.GammaCountExpert(1.0, 2.0))
    # mean(LRMoE.GammaCountExpert(2.0, 2.0))
    # mean(LRMoE.GammaCountExpert(3.0, 2.0))
    # mean(LRMoE.GammaCountExpert(4.0, 2.0))
    # mean(LRMoE.GammaCountExpert(5.0, 2.0))

    # mean(LRMoE.GammaCountExpert(1.0, 3.0))
    # mean(LRMoE.GammaCountExpert(2.0, 3.0))
    # mean(LRMoE.GammaCountExpert(3.0, 3.0))
    # mean(LRMoE.GammaCountExpert(4.0, 3.0))
    # mean(LRMoE.GammaCountExpert(5.0, 3.0))

    # mean(LRMoE.GammaCountExpert(1.0, 1/1.0))
    # mean(LRMoE.GammaCountExpert(1.0, 1/2.0))
    # mean(LRMoE.GammaCountExpert(1.0, 1/3.0))
    # mean(LRMoE.GammaCountExpert(1.0, 1/4.0))
    # mean(LRMoE.GammaCountExpert(1.0, 1/5.0))
    # mean(LRMoE.GammaCountExpert(1.0, 1/6.0))
    # mean(LRMoE.GammaCountExpert(1.0, 1/7.0))
    # mean(LRMoE.GammaCountExpert(1.0, 1/8.0))
    # mean(LRMoE.GammaCountExpert(1.0, 1/9.0))

    # var(LRMoE.GammaCountExpert(1.0, 1/1.0))
    # var(LRMoE.GammaCountExpert(1.0, 1/2.0))
    # var(LRMoE.GammaCountExpert(1.0, 1/3.0))
    # var(LRMoE.GammaCountExpert(1.0, 1/4.0))
    # var(LRMoE.GammaCountExpert(1.0, 1/5.0))
    # var(LRMoE.GammaCountExpert(1.0, 1/6.0))
    # var(LRMoE.GammaCountExpert(1.0, 1/7.0))
    # var(LRMoE.GammaCountExpert(1.0, 1/8.0))
    # var(LRMoE.GammaCountExpert(1.0, 1/9.0))

    # mean(LRMoE.GammaCountExpert(2.0, 1/1.0))
    # mean(LRMoE.GammaCountExpert(2.0, 1/2.0))
    # mean(LRMoE.GammaCountExpert(2.0, 1/3.0))
    # mean(LRMoE.GammaCountExpert(2.0, 1/4.0))
    # mean(LRMoE.GammaCountExpert(2.0, 1/5.0))
    # mean(LRMoE.GammaCountExpert(2.0, 1/6.0))
    # mean(LRMoE.GammaCountExpert(2.0, 1/7.0))
    # mean(LRMoE.GammaCountExpert(2.0, 1/8.0))
    # mean(LRMoE.GammaCountExpert(2.0, 1/9.0))

    # var(LRMoE.GammaCountExpert(2.0, 1/1.0))
    # var(LRMoE.GammaCountExpert(2.0, 1/2.0))
    # var(LRMoE.GammaCountExpert(2.0, 1/3.0))
    # var(LRMoE.GammaCountExpert(2.0, 1/4.0))
    # var(LRMoE.GammaCountExpert(2.0, 1/5.0))
    # var(LRMoE.GammaCountExpert(2.0, 1/6.0))
    # var(LRMoE.GammaCountExpert(2.0, 1/7.0))
    # var(LRMoE.GammaCountExpert(2.0, 1/8.0))
    # var(LRMoE.GammaCountExpert(2.0, 1/9.0))

    # mean(LRMoE.NegativeBinomialExpert(1.0, 0.20))
    # mean(LRMoE.NegativeBinomialExpert(2.0, 0.20))
    # mean(LRMoE.NegativeBinomialExpert(3.0, 0.20))
    # mean(LRMoE.NegativeBinomialExpert(4.0, 0.20))
    # mean(LRMoE.NegativeBinomialExpert(5.0, 0.20))
    # mean(LRMoE.NegativeBinomialExpert(6.0, 0.20))
    # mean(LRMoE.NegativeBinomialExpert(7.0, 0.20))
    # mean(LRMoE.NegativeBinomialExpert(8.0, 0.20))
    # mean(LRMoE.NegativeBinomialExpert(9.0, 0.20))

    # var(LRMoE.NegativeBinomialExpert(1.0, 0.20))
    # var(LRMoE.NegativeBinomialExpert(2.0, 0.20))
    # var(LRMoE.NegativeBinomialExpert(3.0, 0.20))
    # var(LRMoE.NegativeBinomialExpert(4.0, 0.20))
    # var(LRMoE.NegativeBinomialExpert(5.0, 0.20))
    # var(LRMoE.NegativeBinomialExpert(6.0, 0.20))
    # var(LRMoE.NegativeBinomialExpert(7.0, 0.20))
    # var(LRMoE.NegativeBinomialExpert(8.0, 0.20))
    # var(LRMoE.NegativeBinomialExpert(9.0, 0.20))

end

@testset "exposurize_expert" begin

    # param0 = rand(Distributions.Uniform(0, 1), 1)[1]
    # param1 = rand(Distributions.Uniform(10, 20), 1)[1]
    # param2 = rand(Distributions.Uniform(0, 1), 1)[1]
    # param3 = rand(Distributions.Uniform(5, 10), 1)[1]

    # expo = rand(Distributions.Uniform(0.5, 20), 1)[1]

    # tmp = LRMoE.BinomialExpert(20, param0)
    # @test tmp == LRMoE.exposurize_expert(tmp, exposure = expo)
    # tmp = LRMoE.ZIBinomialExpert(0.20, 20, param0)
    # @test tmp == LRMoE.exposurize_expert(tmp, exposure = expo)

    # tmp = LRMoE.GammaCountExpert(param1, param2)
    # @test tmp != LRMoE.exposurize_expert(tmp, exposure = expo)
    # @test isapprox([LRMoE.params(tmp)...] .* [1, 1/expo], [LRMoE.params(LRMoE.exposurize_expert(tmp, exposure = expo))...], atol = 1e-6)
    # tmp = LRMoE.ZIGammaCountExpert(param0, param1, param2)
    # @test tmp != LRMoE.exposurize_expert(tmp, exposure = expo)
    # @test isapprox([LRMoE.params(tmp)...] .* [1, 1, 1/expo], [LRMoE.params(LRMoE.exposurize_expert(tmp, exposure = expo))...], atol = 1e-6)

    # tmp = LRMoE.NegativeBinomialExpert(param1, param0)
    # @test tmp != LRMoE.exposurize_expert(tmp, exposure = expo)
    # @test [LRMoE.params(tmp)...] .* [expo, 1] == [LRMoE.params(LRMoE.exposurize_expert(tmp, exposure = expo))...]
    # tmp = LRMoE.ZINegativeBinomialExpert(param0, param1, param0)
    # @test tmp != LRMoE.exposurize_expert(tmp, exposure = expo)
    # @test [LRMoE.params(tmp)...] .* [1, expo, 1] == [LRMoE.params(LRMoE.exposurize_expert(tmp, exposure = expo))...]

    # tmp = LRMoE.PoissonExpert(param1)
    # @test tmp != LRMoE.exposurize_expert(tmp, exposure = expo)
    # @test [LRMoE.params(tmp)...] .* [expo] == [LRMoE.params(LRMoE.exposurize_expert(tmp, exposure = expo))...]
    # tmp = LRMoE.ZIPoissonExpert(param0, param1)
    # @test tmp != LRMoE.exposurize_expert(tmp, exposure = expo)
    # @test [LRMoE.params(tmp)...] .* [1, expo] == [LRMoE.params(LRMoE.exposurize_expert(tmp, exposure = expo))...]

    # model = [LogNormalExpert(param1, param2) ZIGammaExpert(param0, param1, param2) WeibullExpert(param1, param2) BurrExpert(param1, param2, param3);
    #         BinomialExpert(20, param0) PoissonExpert(param1) ZIGammaCountExpert(param0, param1, param2) ZINegativeBinomialExpert(param0, param1, param0)]

    # expo = rand(Distributions.Uniform(0.5, 20), 15)

    # @test size(LRMoE.exposurize_model(model, exposure = expo)) == (2, 4, 15)

    # @test mean.(LRMoE.exposurize_model(model, exposure = expo))[1,1,:] == fill(mean(model[1,1]), 15)
    # @test mean.(LRMoE.exposurize_model(model, exposure = expo))[1,2,:] == fill(mean(model[1,2]), 15)
    # @test mean.(LRMoE.exposurize_model(model, exposure = expo))[1,3,:] == fill(mean(model[1,3]), 15)
    # @test mean.(LRMoE.exposurize_model(model, exposure = expo))[1,4,:] == fill(mean(model[1,4]), 15)

    # @test mean.(LRMoE.exposurize_model(model, exposure = expo))[2,1,:] == fill(mean(model[2,1]), 15)
    # @test mean.(LRMoE.exposurize_model(model, exposure = expo))[2,2,:] == mean(model[2,2]) .* expo

    # model_expod = LRMoE.exposurize_model(model, exposure = expo)
    # Y_tmp = hcat(rand(Distributions.LogNormal(2, 1), 15), rand(Distributions.Poisson(5), 15))

    # @test size(LRMoE.expert_ll_ind_mat_exact(Y_tmp[1,:], model_expod[:,:,1])) == (2, 4)
    # @test size(LRMoE.expert_ll_ind_mat_exact(Y_tmp[5,:], model_expod[:,:,5])) == (2, 4)
    # @test size(LRMoE.expert_ll_ind_mat_exact(Y_tmp[12,:], model_expod[:,:,12])) == (2, 4)
    # @test size(LRMoE.expert_ll_list_exact(Y_tmp, model_expod)) == (2, 4, 15)

    # idx = rand(1:15, 1)[1]
    # mat_tmp = LRMoE.expert_ll_ind_mat_exact(Y_tmp[idx,:], model_expod[:,:,idx])
    # @test mat_tmp[1,1] == LRMoE.expert_ll_exact(model_expod[1,1,idx], Y_tmp[idx,1])
    # @test mat_tmp[1,2] == LRMoE.expert_ll_exact(model_expod[1,2,idx], Y_tmp[idx,1])
    # @test mat_tmp[1,3] == LRMoE.expert_ll_exact(model_expod[1,3,idx], Y_tmp[idx,1])
    # @test mat_tmp[1,4] == LRMoE.expert_ll_exact(model_expod[1,4,idx], Y_tmp[idx,1])
    # @test mat_tmp[2,1] == LRMoE.expert_ll_exact(model_expod[2,1,idx], Y_tmp[idx,2])
    # @test mat_tmp[2,2] == LRMoE.expert_ll_exact(model_expod[2,2,idx], Y_tmp[idx,2])
    # @test mat_tmp[2,3] == LRMoE.expert_ll_exact(model_expod[2,3,idx], Y_tmp[idx,2])
    # @test mat_tmp[2,3] == LRMoE.expert_ll_exact(model_expod[2,3,idx], Y_tmp[idx,2])

    # idx = rand(1:15, 1)[1]
    # cube_tmp = LRMoE.expert_ll_list_exact(Y_tmp, model_expod)
    # @test cube_tmp[1,1,idx] == LRMoE.expert_ll_exact(model_expod[1,1,idx], Y_tmp[idx,1])
    # @test cube_tmp[1,2,idx] == LRMoE.expert_ll_exact(model_expod[1,2,idx], Y_tmp[idx,1])
    # @test cube_tmp[1,3,idx] == LRMoE.expert_ll_exact(model_expod[1,3,idx], Y_tmp[idx,1])
    # @test cube_tmp[1,4,idx] == LRMoE.expert_ll_exact(model_expod[1,4,idx], Y_tmp[idx,1])
    # @test cube_tmp[2,1,idx] == LRMoE.expert_ll_exact(model_expod[2,1,idx], Y_tmp[idx,2])
    # @test cube_tmp[2,2,idx] == LRMoE.expert_ll_exact(model_expod[2,2,idx], Y_tmp[idx,2])
    # @test cube_tmp[2,3,idx] == LRMoE.expert_ll_exact(model_expod[2,3,idx], Y_tmp[idx,2])
    # @test cube_tmp[2,4,idx] == LRMoE.expert_ll_exact(model_expod[2,4,idx], Y_tmp[idx,2])

end

@testset "exposurize_model" begin
    # param0 = rand(Distributions.Uniform(0, 1), 1)[1]
    # param1 = rand(Distributions.Uniform(10, 20), 1)[1]
    # param2 = rand(Distributions.Uniform(0, 1), 1)[1]
    # param3 = rand(Distributions.Uniform(5, 10), 1)[1]

    # model = [LogNormalExpert(param1, param2) ZIGammaExpert(param0, param1, param2) WeibullExpert(param1, param2) BurrExpert(param1, param2, param3);
    #         BinomialExpert(20, param0) PoissonExpert(param1) ZIGammaCountExpert(param0, param1, param2) ZINegativeBinomialExpert(param0, param1, param0)]

    # expo = rand(Distributions.Uniform(0.5, 20), 15)

    # @test size(LRMoE.exposurize_model(model, exposure = expo)) == (2, 4, 15)

    # @test mean.(LRMoE.exposurize_model(model, exposure = expo))[1,1,:] == fill(mean(model[1,1]), 15)
    # @test mean.(LRMoE.exposurize_model(model, exposure = expo))[1,2,:] == fill(mean(model[1,2]), 15)
    # @test mean.(LRMoE.exposurize_model(model, exposure = expo))[1,3,:] == fill(mean(model[1,3]), 15)
    # @test mean.(LRMoE.exposurize_model(model, exposure = expo))[1,4,:] == fill(mean(model[1,4]), 15)

    # @test mean.(LRMoE.exposurize_model(model, exposure = expo))[2,1,:] == fill(mean(model[2,1]), 15)
    # @test mean.(LRMoE.exposurize_model(model, exposure = expo))[2,2,:] == mean(model[2,2]) .* expo

    # model_expod = LRMoE.exposurize_model(model, exposure = expo)
    # Y_tmp = hcat(rand(Distributions.LogNormal(2, 1), 15), rand(Distributions.Poisson(5), 15))

    # @test size(LRMoE.expert_ll_ind_mat_exact(Y_tmp[1,:], model_expod[:,:,1])) == (2, 4)
    # @test size(LRMoE.expert_ll_ind_mat_exact(Y_tmp[5,:], model_expod[:,:,5])) == (2, 4)
    # @test size(LRMoE.expert_ll_ind_mat_exact(Y_tmp[12,:], model_expod[:,:,12])) == (2, 4)
    # @test size(LRMoE.expert_ll_list_exact(Y_tmp, model_expod)) == (2, 4, 15)

    # idx = rand(1:15, 1)[1]
    # mat_tmp = LRMoE.expert_ll_ind_mat_exact(Y_tmp[idx,:], model_expod[:,:,idx])
    # @test mat_tmp[1,1] == LRMoE.expert_ll_exact(model_expod[1,1,idx], Y_tmp[idx,1])
    # @test mat_tmp[1,2] == LRMoE.expert_ll_exact(model_expod[1,2,idx], Y_tmp[idx,1])
    # @test mat_tmp[1,3] == LRMoE.expert_ll_exact(model_expod[1,3,idx], Y_tmp[idx,1])
    # @test mat_tmp[1,4] == LRMoE.expert_ll_exact(model_expod[1,4,idx], Y_tmp[idx,1])
    # @test mat_tmp[2,1] == LRMoE.expert_ll_exact(model_expod[2,1,idx], Y_tmp[idx,2])
    # @test mat_tmp[2,2] == LRMoE.expert_ll_exact(model_expod[2,2,idx], Y_tmp[idx,2])
    # @test mat_tmp[2,3] == LRMoE.expert_ll_exact(model_expod[2,3,idx], Y_tmp[idx,2])
    # @test mat_tmp[2,3] == LRMoE.expert_ll_exact(model_expod[2,3,idx], Y_tmp[idx,2])

    # idx = rand(1:15, 1)[1]
    # cube_tmp = LRMoE.expert_ll_list_exact(Y_tmp, model_expod)
    # @test cube_tmp[1,1,idx] == LRMoE.expert_ll_exact(model_expod[1,1,idx], Y_tmp[idx,1])
    # @test cube_tmp[1,2,idx] == LRMoE.expert_ll_exact(model_expod[1,2,idx], Y_tmp[idx,1])
    # @test cube_tmp[1,3,idx] == LRMoE.expert_ll_exact(model_expod[1,3,idx], Y_tmp[idx,1])
    # @test cube_tmp[1,4,idx] == LRMoE.expert_ll_exact(model_expod[1,4,idx], Y_tmp[idx,1])
    # @test cube_tmp[2,1,idx] == LRMoE.expert_ll_exact(model_expod[2,1,idx], Y_tmp[idx,2])
    # @test cube_tmp[2,2,idx] == LRMoE.expert_ll_exact(model_expod[2,2,idx], Y_tmp[idx,2])
    # @test cube_tmp[2,3,idx] == LRMoE.expert_ll_exact(model_expod[2,3,idx], Y_tmp[idx,2])
    # @test cube_tmp[2,4,idx] == LRMoE.expert_ll_exact(model_expod[2,4,idx], Y_tmp[idx,2])

    # Y_tmp = hcat(0.2 .* Y_tmp[:,1], 0.8 .* Y_tmp[:,1], 1.2 .* Y_tmp[:,1], 5 .* Y_tmp[:,1], fill(0, length(Y_tmp[:,2])), Y_tmp[:,2], Y_tmp[:,2], fill(Inf, length(Y_tmp[:,2])))

    # @test size(LRMoE.expert_ll_ind_mat(Y_tmp[1,:], model_expod[:,:,1])) == (2, 4)
    # @test size(LRMoE.expert_ll_ind_mat(Y_tmp[5,:], model_expod[:,:,5])) == (2, 4)
    # @test size(LRMoE.expert_ll_ind_mat(Y_tmp[12,:], model_expod[:,:,12])) == (2, 4)
    # @test size(LRMoE.expert_ll_list(Y_tmp, model_expod)) == (2, 4, 15)

    # idx = rand(1:15, 1)[1]
    # mat_tmp = LRMoE.expert_ll_ind_mat(Y_tmp[idx,:], model_expod[:,:,idx])
    # @test mat_tmp[1,1] == LRMoE.expert_ll(model_expod[1,1,idx], Y_tmp[idx,1], Y_tmp[idx,2], Y_tmp[idx,3], Y_tmp[idx,4])
    # @test mat_tmp[1,2] == LRMoE.expert_ll(model_expod[1,2,idx], Y_tmp[idx,1], Y_tmp[idx,2], Y_tmp[idx,3], Y_tmp[idx,4])
    # @test mat_tmp[1,3] == LRMoE.expert_ll(model_expod[1,3,idx], Y_tmp[idx,1], Y_tmp[idx,2], Y_tmp[idx,3], Y_tmp[idx,4])
    # @test mat_tmp[1,4] == LRMoE.expert_ll(model_expod[1,4,idx], Y_tmp[idx,1], Y_tmp[idx,2], Y_tmp[idx,3], Y_tmp[idx,4])
    # @test mat_tmp[2,1] == LRMoE.expert_ll(model_expod[2,1,idx], Y_tmp[idx,5], Y_tmp[idx,6], Y_tmp[idx,7], Y_tmp[idx,8])
    # @test mat_tmp[2,2] == LRMoE.expert_ll(model_expod[2,2,idx], Y_tmp[idx,5], Y_tmp[idx,6], Y_tmp[idx,7], Y_tmp[idx,8])
    # @test mat_tmp[2,3] == LRMoE.expert_ll(model_expod[2,3,idx], Y_tmp[idx,5], Y_tmp[idx,6], Y_tmp[idx,7], Y_tmp[idx,8])
    # @test mat_tmp[2,3] == LRMoE.expert_ll(model_expod[2,3,idx], Y_tmp[idx,5], Y_tmp[idx,6], Y_tmp[idx,7], Y_tmp[idx,8])

    # idx = rand(1:15, 1)[1]
    # cube_tmp = LRMoE.expert_ll_list(Y_tmp, model_expod)
    # @test cube_tmp[1,1,idx] == LRMoE.expert_ll(model_expod[1,1,idx], Y_tmp[idx,1], Y_tmp[idx,2], Y_tmp[idx,3], Y_tmp[idx,4])
    # @test cube_tmp[1,2,idx] == LRMoE.expert_ll(model_expod[1,2,idx], Y_tmp[idx,1], Y_tmp[idx,2], Y_tmp[idx,3], Y_tmp[idx,4])
    # @test cube_tmp[1,3,idx] == LRMoE.expert_ll(model_expod[1,3,idx], Y_tmp[idx,1], Y_tmp[idx,2], Y_tmp[idx,3], Y_tmp[idx,4])
    # @test cube_tmp[1,4,idx] == LRMoE.expert_ll(model_expod[1,4,idx], Y_tmp[idx,1], Y_tmp[idx,2], Y_tmp[idx,3], Y_tmp[idx,4])
    # @test cube_tmp[2,1,idx] == LRMoE.expert_ll(model_expod[2,1,idx], Y_tmp[idx,5], Y_tmp[idx,6], Y_tmp[idx,7], Y_tmp[idx,8])
    # @test cube_tmp[2,2,idx] == LRMoE.expert_ll(model_expod[2,2,idx], Y_tmp[idx,5], Y_tmp[idx,6], Y_tmp[idx,7], Y_tmp[idx,8])
    # @test cube_tmp[2,3,idx] == LRMoE.expert_ll(model_expod[2,3,idx], Y_tmp[idx,5], Y_tmp[idx,6], Y_tmp[idx,7], Y_tmp[idx,8])
    # @test cube_tmp[2,4,idx] == LRMoE.expert_ll(model_expod[2,4,idx], Y_tmp[idx,5], Y_tmp[idx,6], Y_tmp[idx,7], Y_tmp[idx,8])

    # # ll related
    # gate_tmp = rand(Distributions.Uniform(-1, 1), 15, 4)

    # dim_agg = LRMoE.loglik_aggre_dim(cube_tmp)
    # @test size(dim_agg) == (15, 4)
    # @test size(LRMoE.loglik_aggre_gate_dim(gate_tmp, dim_agg)) == (15, 4)

    # # Simulation related
    # # X_tmp = rand(Distributions.Uniform(-1, 1), 15, 7)
    # # α_tmp = rand(Distributions.Uniform(-1, 1), 4, 7)

    # # model = [LogNormalExpert(param1, param2) ZIGammaExpert(param0, param1, param2) WeibullExpert(param1, param2) BurrExpert(param1, param2, param3);
    # #         BinomialExpert(20, param0) PoissonExpert(param1) ZIGammaCountExpert(param0, param1, param2) ZINegativeBinomialExpert(param0, param1, param0)]

    # # expo = rand(Distributions.Uniform(0.5, 20), 15)

    # # model_expod = LRMoE.exposurize_model(model, exposure = expo)

    # # sim_tmp = LRMoE.sim_dataset(α_tmp, X_tmp, model, exposure = expo)
    # # mean(sim_tmp, dims = 1)

    # # sim_tmp = LRMoE.sim_dataset(α_tmp, X_tmp, model)
    # # mean(sim_tmp, dims = 1)

end