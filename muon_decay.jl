#region Imports
using Pkg
Pkg.activate(".")
using SymbolicRegression
using CSV
using DataFrames
using Random
using StatsBase
using IterTools
include("./config_management/muon_decay_config.jl")
using Base.Threads
using Dates
using FilePathsBase
using Serialization
# using Plots
# gr()
cfg_data=CONFIG_data
cfg_log=CONFIG_log
cfg_sr=CONFIG_sr

timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
log_dir = "logs/" * cfg_log["log_folder_prefix"] * "_log_$timestamp"
if !isdir(log_dir)
    mkpath(log_dir)
end

#endregion

# println("Imports Completed!...Press any key to continue...")
# readline()


#region Load the data
max_marginal_index = cfg_data["num_dimensions"]-1
max_conditional_index = cfg_data["num_conditional_slices"]-1
df_m = [CSV.read(cfg_data["data_path_and_prefix"] * "_marginal_data_$(i).csv", DataFrame; header=false) for i in 0:max_marginal_index]
df_c_slices = [CSV.read(cfg_data["data_path_and_prefix"] * "_conditional_slices_$(i).csv", DataFrame; header=false) for i in 0:max_marginal_index]
df_c_data_slices = [[CSV.read(cfg_data["data_path_and_prefix"] * "_conditional_data_$(i)_slice_$(j).csv", DataFrame; header=false) for j in 0:max_conditional_index] for i in 0:max_marginal_index]

m_xd = [df[:, 1] for df in df_m]
m_yd = [df[:, end] for df in df_m]

num_variables_conditioned_on = cfg_data["num_dimensions"]-1
c_xd_slice_info = [[collect(df_c_slices[d][i, 1:num_variables_conditioned_on]) for i in 1:cfg_data["num_conditional_slices"]] for d in 1:cfg_data["num_dimensions"]]
c_yd_slice_info = [[df_c_slices[d][i, end] for i in 1:cfg_data["num_conditional_slices"]] for d in 1:cfg_data["num_dimensions"]]

c_xd = [[df[:, 1] for df in df_c_data_slices[d]] for d in 1:cfg_data["num_dimensions"]]
c_yd = [[df[:, 2] for df in df_c_data_slices[d]] for d in 1:cfg_data["num_dimensions"]]

joint_data_x = vcat([vcat([hcat(x, repeat(info', length(x))) for (x, info) in zip(c_xd[d], c_xd_slice_info[d])]...) for d in 1:cfg_data["num_dimensions"]]...)

joint_data_y = vcat([vcat([y*info for (y, info) in zip(c_yd[1], c_yd_slice_info[1])]...) for d in 1:cfg_data["num_dimensions"]]...)
#endregion

# println("Data Loaded!...Press any key to continue...")
# readline()


#region Helper function to update the feature of a node in the tree
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
#endregion


#region Set options for Marginal and Conditional SR 
options = SymbolicRegression.Options(;
    binary_operators=cfg_sr["binary_operators"], unary_operators=cfg_sr["unary_operators"]
)
#endregion


#region Marginal SR calls
marginal_halls_of_fame = Vector{Any}(undef, cfg_data["num_dimensions"])
dominating_pareto_marginals = Vector{Any}(undef, cfg_data["num_dimensions"])
trees_marginals = Vector{Any}(undef, cfg_data["num_dimensions"])
@threads for d in 1:cfg_data["num_dimensions"]

    #region marginal_log
        hall_of_fame = equation_search(
            reshape(m_xd[d], 1, :), m_yd[d]; 
            options=options, 
            parallelism=cfg_sr["parallelism_for_marginal_sr"], 
            niterations=cfg_sr["niterations_for_marginal_sr"]
        )

        pareto = calculate_pareto_frontier(hall_of_fame)
        trees = [member.tree for member in pareto]

        for member in eachindex(hall_of_fame.members)
            update_feature!(hall_of_fame.members[member].tree.tree, 1, d)
        end
        
        marginal_halls_of_fame[d] = hall_of_fame
        dominating_pareto_marginals[d] = pareto
        trees_marginals[d] = trees
end
#endregion

# println("Marginal SR Completed!...Press any key to continue...")
# readline()


#region Conditional SR calls
conditional_halls_of_fame_per_slice = Vector{Any}(undef, cfg_data["num_conditional_slices"])
conditional_halls_of_fame = [conditional_halls_of_fame_per_slice for i in 1:cfg_data["num_dimensions"]]
dominating_pareto_conditionals_per_slice = Vector{Any}(undef, cfg_data["num_conditional_slices"])
dominating_pareto_conditionals = [dominating_pareto_conditionals_per_slice for i in 1:cfg_data["num_dimensions"]]
trees_conditionals_per_slice = Vector{Any}(undef, cfg_data["num_conditional_slices"])
trees_conditionals = [trees_conditionals_per_slice for i in 1:cfg_data["num_dimensions"]]

d_slice_permutations = [(d, slice) for d in 1:cfg_data["num_dimensions"] for slice in 1:cfg_data["num_conditional_slices"]]

@threads for (d, slice) in d_slice_permutations
    x = reshape(c_xd[d][slice], 1, :)
    y = c_yd[d][slice]

        hall_of_fame = equation_search(
            x, y; 
            options=options, 
            parallelism=cfg_sr["parallelism_for_conditional_sr"], 
            niterations=cfg_sr["niterations_for_conditional_sr"],
        )
        pareto = calculate_pareto_frontier(hall_of_fame)
        trees = [member.tree for member in pareto]

        for member in eachindex(hall_of_fame.members)
            update_feature!(hall_of_fame.members[member].tree.tree, 1, d)
        end

        conditional_halls_of_fame[d][slice] = hall_of_fame
        dominating_pareto_conditionals[d][slice] = pareto
        trees_conditionals[d][slice] = trees
end
#endregion

# println("Conditional SR Completed!...Press any key to continue...")
# readline()

#region Joint SR call

#region helper funtions to produce joint expression trees
function one_to_one_multiply_conditionals_with_marginals(conditional_pop_members, marginal_pop_members)
    joint_pop_members = deepcopy(conditional_pop_members)
    for i in eachindex(joint_pop_members)
        joint_pop_members[i].tree = joint_pop_members[i].tree * rand(marginal_pop_members).tree
    end
    return joint_pop_members
end

function cartesian_product_multiply_conditionals_with_marginals(conditional_pop_members, marginal_pop_members)
    joint_pop_members = repeat(deepcopy(conditional_pop_members), inner=size(marginal_pop_members))
    marginal_pop_members_expanded = repeat(deepcopy(marginal_pop_members), outer=size(conditional_pop_members))
    marginal_trees = [member.tree for member in marginal_pop_members_expanded]
    multiply_trees = (conditional_tree, marginal_tree) -> (conditional_tree * marginal_tree)
    setfield!.(joint_pop_members, :tree, multiply_trees.(getfield.(joint_pop_members, :tree), marginal_trees))
    return joint_pop_members
end

if cfg_sr["joint_expression_possibilities"] == "one_to_one"
    multiply_conditionals_with_marginals = one_to_one_multiply_conditionals_with_marginals
elseif cfg_sr["joint_expression_possibilities"] == "cartesian"
    multiply_conditionals_with_marginals = cartesian_product_multiply_conditionals_with_marginals
else
    error("Invalid compute mode: $(cfg_sr["joint_expression_possibilities"])")
end
#endregion

joint_initial_population = []
dimensions = 1:cfg_data["num_dimensions"]
for (d, slice) in d_slice_permutations
    fixed_variables = filter(x -> x !=d, dimensions)
    joint_pop_members_per_dim_and_slice = deepcopy(conditional_halls_of_fame[d][slice].members)
    # This assumes that the marginals are all independent
    for fixed_variable in fixed_variables
       joint_pop_members_per_dim_and_slice = multiply_conditionals_with_marginals(joint_pop_members_per_dim_and_slice, marginal_halls_of_fame[fixed_variable].members)
       if cfg_sr["joint_max_num_expressions_per_dim_and_slice"]!= Inf
            shuffle!(joint_pop_members_per_dim_and_slice)    
            joint_pop_members_per_dim_and_slice = sample(joint_pop_members_per_dim_and_slice, min(length(joint_pop_members_per_dim_and_slice), cfg_sr["joint_max_num_expressions_per_dim_and_slice"]); replace=false)
       end
    end
    append!(joint_initial_population, joint_pop_members_per_dim_and_slice)
end

shuffle!(joint_initial_population)
if cfg_sr["joint_max_num_expressions"] != Inf
    joint_initial_population = sample(joint_initial_population, min(length(joint_initial_population), cfg_sr["joint_max_num_expressions"]); replace=false)
end

populations = [joint_initial_population[i:i+(cfg_sr["population_size_for_joint_sr"]-1)] for i in 1:cfg_sr["population_size_for_joint_sr"]:(cfg_sr["num_populations_for_joint_sr"]*cfg_sr["population_size_for_joint_sr"])]

joint_options = SymbolicRegression.Options(;
    binary_operators=cfg_sr["binary_operators"], unary_operators=cfg_sr["unary_operators"], populations = cfg_sr["num_populations_for_joint_sr"], population_size = cfg_sr["population_size_for_joint_sr"]
    )

# println("Starting joint SR call...Press any key to continue...")
# readline()

    joint_hall_of_fame = equation_search(
            reshape(joint_data_x, cfg_data["num_dimensions"], :), 
            joint_data_y; 
            options=joint_options,
            parallelism=cfg_sr["parallelism_for_joint_sr"], 
            initial_populations=populations, 
            niterations=cfg_sr["niterations_for_joint_sr"], 
    )
#endregion

# Save the joint hall of fame
save_path = joinpath(log_dir, "joint_hall_of_fame.jls")
open(save_path, "w") do io
    serialize(io, joint_hall_of_fame) # to load the data again use deserialze
end

println("Joint Hall of Fame saved to $(save_path)")

println("Joint SR Completed!")