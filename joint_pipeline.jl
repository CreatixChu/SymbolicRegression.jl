using Pkg
Pkg.activate(".")
using SymbolicRegression
using CSV
using DataFrames
using Random
using StatsBase
using IterTools
include("./config_management/cluster_toy_polynomial_config.jl")
using Base.Threads
using LoggingExtras
using Dates
using FilePathsBase
using Serialization
cfg_data = CONFIG_data
cfg_log = CONFIG_log
cfg_sr = CONFIG_sr

top_level_log_dir = "./logs"

log_dir = joinpath(top_level_log_dir, maximum(filter(f -> startswith(f, "cluster_"), readdir(top_level_log_dir))))

max_marginal_index = cfg_data["num_dimensions"] - 1
max_conditional_index = cfg_data["num_conditional_slices"] - 1


df_m = [
    CSV.read(
        cfg_data["data_path_and_prefix"] * "_marginal_data_$(i).csv",
        DataFrame;
        header=false,
    ) for i in 0:max_marginal_index
]

df_c_slices = [
    CSV.read(
        cfg_data["data_path_and_prefix"] * "_conditional_slices_$(i).csv",
        DataFrame;
        header=false,
    ) for i in 0:max_marginal_index
]

df_c_data_slices = [
    [
        CSV.read(
            cfg_data["data_path_and_prefix"] * "_conditional_data_$(i)_slice_$(j).csv",
            DataFrame;
            header=false,
        ) for j in 0:max_conditional_index
    ] for i in 0:max_marginal_index
]

m_xd = [df[:, 1] for df in df_m]
m_yd = [df[:, end] for df in df_m]

num_variables_conditioned_on = cfg_data["num_dimensions"] - 1
c_xd_slice_info = [
    [
        collect(df_c_slices[d][i, 1:num_variables_conditioned_on]) for
        i in 1:cfg_data["num_conditional_slices"]
    ] for d in 1:cfg_data["num_dimensions"]
]
c_yd_slice_info = [
    [df_c_slices[d][i, end] for i in 1:cfg_data["num_conditional_slices"]] for
    d in 1:cfg_data["num_dimensions"]
]

c_xd = [[df[:, 1] for df in df_c_data_slices[d]] for d in 1:cfg_data["num_dimensions"]]
c_yd = [[df[:, 2] for df in df_c_data_slices[d]] for d in 1:cfg_data["num_dimensions"]]

joint_data_x = vcat(
    [
        vcat(
            [
                hcat(x, repeat(info', length(x))) for
                (x, info) in zip(c_xd[d], c_xd_slice_info[d])
            ]...,
        ) for d in 1:cfg_data["num_dimensions"]
    ]...,
)

joint_data_y = vcat(
    [
        vcat([y * info for (y, info) in zip(c_yd[1], c_yd_slice_info[1])]...) for
        d in 1:cfg_data["num_dimensions"]
    ]...,
)

d_slice_permutations = [
    (d, slice) for d in 1:cfg_data["num_dimensions"] for
    slice in 1:cfg_data["num_conditional_slices"]
]

joint_options = SymbolicRegression.Options(;
    binary_operators=cfg_sr["binary_operators"],
    unary_operators=cfg_sr["unary_operators"],
    populations=cfg_sr["num_populations_for_joint_sr"],
    population_size=cfg_sr["population_size_for_joint_sr"],
    constraints=cfg_sr["constraints"],
    nested_constraints=cfg_sr["nested_constraints"],
    output_directory=log_dir,
    maxsize=cfg_sr["maxsize"],
    ncycles_per_iteration=cfg_sr["ncycles_per_iteration"],
    parsimony=cfg_sr["parsimony"],
    warmup_maxsize_by=cfg_sr["warmup_maxsize_by"],
    adaptive_parsimony_scaling=cfg_sr["adaptive_parsimony_scaling"],
    progress=cfg_sr["progress"],
    verbosity=cfg_sr["verbosity"]
)

joint_options_no_init = SymbolicRegression.Options(;
    binary_operators=cfg_sr["binary_operators"],
    unary_operators=cfg_sr["unary_operators"],
    populations=cfg_sr["num_populations_for_joint_sr"],
    population_size=cfg_sr["population_size_for_joint_sr"],
    constraints=cfg_sr["constraints"],
    nested_constraints=cfg_sr["nested_constraints"],
    output_directory=log_dir * "/joint_no_init",
    maxsize=cfg_sr["maxsize"],
    ncycles_per_iteration=cfg_sr["ncycles_per_iteration"],
    parsimony=cfg_sr["parsimony"],
    warmup_maxsize_by=cfg_sr["warmup_maxsize_by"],
    adaptive_parsimony_scaling=cfg_sr["adaptive_parsimony_scaling"],
    progress=cfg_sr["progress"],
    verbosity=cfg_sr["verbosity"]
)

# Save the marginal and conditional hall of fame and pareto frontiers
read_path = joinpath(log_dir, "marginal_halls_of_fame.jls")
marginal_halls_of_fame = deserialize(read_path)

read_path = joinpath(log_dir, "dominating_pareto_marginals.jls")
dominating_pareto_marginals = deserialize(read_path)

read_path = joinpath(log_dir, "conditional_halls_of_fame.jls")
conditional_halls_of_fame = deserialize(read_path)

read_path = joinpath(log_dir, "dominating_pareto_conditionals.jls")
dominating_pareto_conditionals = deserialize(read_path)

#region Joint SR call

#region helper funtions to produce joint expression trees
function one_to_one_multiply_conditionals_with_marginals(
    conditional_pop_members, marginal_pop_members
)
    joint_pop_members = deepcopy(conditional_pop_members)
    for i in eachindex(joint_pop_members)
        joint_pop_members[i].tree =
            joint_pop_members[i].tree * rand(marginal_pop_members).tree
    end
    return joint_pop_members
end

function cartesian_product_multiply_conditionals_with_marginals(
    conditional_pop_members, marginal_pop_members
)
    joint_pop_members = repeat(
        deepcopy(conditional_pop_members); inner=size(marginal_pop_members)
    )
    joint_pop_members = [deepcopy(member) for member in joint_pop_members]
    marginal_pop_members_expanded = repeat(
        deepcopy(marginal_pop_members); outer=size(conditional_pop_members)
    )
    marginal_pop_members_expanded = [
        deepcopy(member) for member in marginal_pop_members_expanded
    ]
    marginal_trees = [member.tree for member in marginal_pop_members_expanded]
    multiply_trees = (conditional_tree, marginal_tree) -> (conditional_tree * marginal_tree)
    setfield!.(
        joint_pop_members,
        :tree,
        multiply_trees.(getfield.(joint_pop_members, :tree), marginal_trees),
    )
    return joint_pop_members
end

if cfg_sr["joint_expression_possibilities"] == "one_to_one"
    multiply_conditionals_with_marginals = one_to_one_multiply_conditionals_with_marginals
elseif cfg_sr["joint_expression_possibilities"] == "cartesian"
    multiply_conditionals_with_marginals =
        cartesian_product_multiply_conditionals_with_marginals
else
    error("Invalid compute mode: $(cfg_sr["joint_expression_possibilities"])")
end
#endregion

println("Setting up initial_population...")
joint_initial_population = []
dimensions = 1:cfg_data["num_dimensions"]
for (d, slice) in d_slice_permutations
    fixed_variables = filter(x -> x != d, dimensions)
    joint_pop_members_per_dim_and_slice = deepcopy(
        conditional_halls_of_fame[d][slice].members
    )
    # This assumes that the marginals are all independent
    for fixed_variable in fixed_variables
        joint_pop_members_per_dim_and_slice = multiply_conditionals_with_marginals(
            joint_pop_members_per_dim_and_slice, dominating_pareto_marginals[fixed_variable]
        )
        if cfg_sr["joint_max_num_expressions_per_dim_and_slice"] != Inf
            shuffle!(joint_pop_members_per_dim_and_slice)
            joint_pop_members_per_dim_and_slice = sample(
                joint_pop_members_per_dim_and_slice,
                min(
                    length(joint_pop_members_per_dim_and_slice),
                    cfg_sr["joint_max_num_expressions_per_dim_and_slice"],
                );
                replace=false,
            )
        end
    end
    append!(joint_initial_population, joint_pop_members_per_dim_and_slice)
end

shuffle!(joint_initial_population)
if cfg_sr["joint_max_num_expressions"] != Inf
    joint_initial_population = sample(
        joint_initial_population,
        min(length(joint_initial_population), cfg_sr["joint_max_num_expressions"]);
        replace=false,
    )
end

println("Alocating expressions to populations...")
populations = [
    joint_initial_population[i:(i + (cfg_sr["population_size_for_joint_sr"] - 1))] for i in
    1:cfg_sr["population_size_for_joint_sr"]:(cfg_sr["num_populations_for_joint_sr"] * cfg_sr["population_size_for_joint_sr"])
]

println("Starting joint SR call!")

joint_hall_of_fame = equation_search(
    reshape(joint_data_x, cfg_data["num_dimensions"], :),
    joint_data_y;
    options=joint_options,
    initial_populations=populations,
    parallelism=cfg_sr["parallelism_for_joint_sr"],
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

println("Starting second joint SR without initialization!")

joint_hall_of_fame = equation_search(
    reshape(joint_data_x, cfg_data["num_dimensions"], :),
    joint_data_y;
    options=joint_options_no_init,
    parallelism=cfg_sr["parallelism_for_joint_sr"],
    niterations=cfg_sr["niterations_for_joint_sr"],
)
