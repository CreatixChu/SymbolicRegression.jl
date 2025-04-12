using Pkg
Pkg.activate(".")
using Wandb
using SymbolicRegression
using CSV
using DataFrames
using Random
using Plots
using Base.Threads
using Dates
using Logging

gr()
include("config.jl")
cfg=CONFIG

lg = WandbLogger(project = "SR.jl",
                 name = "SR-$(now())",
                 config = cfg)
global_logger(lg)


Wandb.log(lg, Dict("accuracy" => 0.9, "loss" => 0.3))
Wandb.log(lg, "hello World")

function update_feature!(node::Node, source_feature::Int, target_feature::Int)
    # Only update leaf (degree==0) feature nodes (non-constant)
    if node.degree == 0 && !node.constant
        if node.feature == source_feature
            node.feature = target_feature
        end
    elseif node.degree >= 1
        # Recursively update children: left child is always defined;
        update_feature!(node.l, source_feature, target_feature)
        # Right child is defined only for binary operators (degree==2)
        if node.degree == 2
            update_feature!(node.r, source_feature, target_feature)
        end
    end
    return node
end

df_m = [CSV.read("./data/marginal_data_$(i).csv", DataFrame; header=false) for i in 0:1]
df_c_slices = [CSV.read("./data/conditional_slices_$(i).csv", DataFrame; header=false) for i in 0:1]
df_c_data_slices = [[CSV.read("./data/conditional_data_$(i)_slice_$(j).csv", DataFrame; header=false) for j in 0:7] for i in 0:1]

m_x = [df[:, 1] for df in df_m]
m_y = [df[:, 2] for df in df_m]

c_x1_slice_info = [df_c_slices[1][i, 1] for i in 1:8]
c_y1_slice_info = [df_c_slices[1][i, 2] for i in 1:8]

c_x2_slice_info = [df_c_slices[2][i, 1] for i in 1:8]
c_y2_slice_info = [df_c_slices[2][i, 2] for i in 1:8]

c_x1 = [df[:, 1] for df in df_c_data_slices[1]]
c_y1 = [df[:, 2] for df in df_c_data_slices[1]]

c_x2 = [df[:, 1] for df in df_c_data_slices[2]]
c_y2 = [df[:, 2] for df in df_c_data_slices[2]]

conditional_data_x1 = [[x] for x in c_x1]
conditional_data_x2 = [[x] for x in c_x2]
conditional_data_y1 = [[y] for y in c_y1]
conditional_data_y2 = [[y] for y in c_y2]

joint_data_x = vcat(
    [hcat(x, repeat([info], length(x))) for (x, info) in zip(c_x1, c_x1_slice_info)]...,
    [hcat(x, repeat([info], length(x))) for (x, info) in zip(c_x2, c_x2_slice_info)]...
)

joint_data_y = vcat(
    [y .* info for (y, info) in zip(c_y1, c_y1_slice_info)]...,
    [y .* info for (y, info) in zip(c_y2, c_y2_slice_info)]...
)


m_x1_p = plot(m_x[1], m_y[1], seriestype=:scatter, title="Scatter plot of data x0", xlabel="X-axis", ylabel="Y-axis")
m_x2_p = plot(m_x[2], m_y[2], seriestype=:scatter, title="Scatter plot of data x1", xlabel="X-axis", ylabel="Y-axis")
# display(m_x1_p)
# display(m_x2_p)

#region Low level API
options = SymbolicRegression.Options(;
    binary_operators=cfg["binary_operators"], unary_operators=cfg["unary_operators"]
)

hall_of_fame_m_x1 = equation_search(
        reshape(m_x[1], 1, :), m_y[1]; options=options, parallelism=cfg["parallelism_for_marginal_sr"], niterations=cfg["niterations_for_marginal_sr"]
    )

hall_of_fame_m_x2 = equation_search(
        reshape(m_x[2], 1, :), m_y[2]; options=options, parallelism=cfg["parallelism_for_marginal_sr"], niterations=cfg["niterations_for_marginal_sr"]
    )

dominating_m_x1 = calculate_pareto_frontier(hall_of_fame_m_x1)
trees_m_x1 = [member.tree for member in dominating_m_x1]

dominating_m_x2 = calculate_pareto_frontier(hall_of_fame_m_x2)
trees_m_x2 = [member.tree for member in dominating_m_x2]


for i in eachindex(hall_of_fame_m_x2.members)
    update_feature!(hall_of_fame_m_x2.members[i].tree.tree, 1, 2)
end

n1 = length(conditional_data_x1)
conditional_hall_of_fame_x1 = Vector{Any}(undef, n1)
dominating_c_x1 = Vector{Any}(undef, n1)
trees_c_x1 = Vector{Any}(undef, n1)

@threads for i in 1:n1
    x = reshape(conditional_data_x1[i][1], 1, :)
    y = conditional_data_y1[i][1]

    hall_of_fame = equation_search(x, y; options=options, parallelism=cfg["parallelism_for_conditional_sr"], niterations=cfg["niterations_for_conditional_sr"])
    conditional_hall_of_fame_x1[i] = hall_of_fame

    pareto = calculate_pareto_frontier(hall_of_fame)
    dominating_c_x1[i] = pareto

    trees_c_x1[i] = [member.tree for member in pareto]
    # @info "metrics" epoch=i
end

n2 = length(conditional_data_x2)
conditional_hall_of_fame_x2 = Vector{Any}(undef, n2)
dominating_c_x2 = Vector{Any}(undef, n2)
trees_c_x2 = Vector{Any}(undef, n2)

@threads for i in 1:n2
    x = reshape(conditional_data_x2[i][1], 1, :)
    y = conditional_data_y2[i][1]

    hall_of_fame = equation_search(x, y; options=options, parallelism=cfg["parallelism_for_conditional_sr"], niterations=cfg["niterations_for_conditional_sr"])
    conditional_hall_of_fame_x2[i] = hall_of_fame

    pareto = calculate_pareto_frontier(hall_of_fame)
    dominating_c_x2[i] = pareto

    trees = [member.tree for member in hall_of_fame.members]
    for tree in trees
        update_feature!(tree.tree, 1, 2)
    end
    trees_c_x2[i] = trees
    # @info "metrics" epoch=i + n1
end

joint_initial_population = []

function multiply_conditionals_with_marginals(conditional_pop_members, marginal_pop_members)
    joint_pop_members = deepcopy(conditional_pop_members)
    for i in eachindex(joint_pop_members)
        joint_pop_members[i].tree = joint_pop_members[i].tree * rand(marginal_pop_members).tree
    end
    return joint_pop_members
end


for i in eachindex(conditional_hall_of_fame_x1)
    append!(joint_initial_population, multiply_conditionals_with_marginals(conditional_hall_of_fame_x1[i].members, hall_of_fame_m_x2.members))
end

for i in eachindex(conditional_hall_of_fame_x2)
    append!(joint_initial_population, multiply_conditionals_with_marginals(conditional_hall_of_fame_x2[i].members, hall_of_fame_m_x1.members))
end

shuffle(joint_initial_population)
# println("Press any key to continue...")
# readline()

populations = [joint_initial_population[i:i+29] for i in 1:30:480]

options1 = SymbolicRegression.Options(;
    binary_operators=cfg["binary_operators"], unary_operators=cfg["unary_operators"], populations = length(populations), population_size = length(populations[1])
    )

# println("Press any key to continue...at end")
# readline()

hof = equation_search(
        reshape(joint_data_x, 2, :), joint_data_y; options=options1, parallelism=:serial, initial_populations=populations, niterations=cfg["niterations_for_joint_sr"]
)

close(lg)






# Bug: something is wrong with the conditional slice probabilities in the dataset!!!!
# Multiply marginals and conditionals to obtain PopMember for initialization
# 30*8 = 240 conditionals
# 30 marginals
# 240*30 = 7200
# 7200*2 = 14,400
