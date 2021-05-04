using Test
using Distributions
using StatsFuns
# using Random


using LRMoE
using JLD2

# Random.seed!(1234)

@load "test/X_all.JLD2" X_all
@load "test/Y_all.JLD2" Y_all

Y_all = convert(Matrix{Int}, Y_all)

# tmp = LRMoE.cmm_init(Y_all, X_all, 3, ["discrete" "discrete"]; exact_Y = true, n_random = 0)

# tmp.params_init[1][1][8]

# 7: ZI-NB
# 8: ZI-GC

model_list = []

for n_comp in 2:20
    init = LRMoE.cmm_init(Y_all, X_all, n_comp, ["discrete" "discrete"]; exact_Y = true, n_random = 0)
    # push!(model_list, (init.α_init, init.ll_best))
    tmp = vcat([hcat([init.params_init[d][i][8] for i in 1:n_comp]...) for d in 1:2]...)
    push!(model_list, (init.α_init, tmp))
end

length(model_list)

model_list[1][2]
model_list[2][2]
model_list[3][2]

@save "model_list_GC.JLD2" model_list