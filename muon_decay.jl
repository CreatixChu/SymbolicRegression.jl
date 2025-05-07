#region Imports
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
cfg_data=CONFIG_data
cfg_log=CONFIG_log
cfg_sr=CONFIG_sr

function format_hms(delta::Period)
    total_seconds = Millisecond(delta).value ÷ 1000
    hours = total_seconds ÷ 3600
    minutes = (total_seconds % 3600) ÷ 60
    seconds = total_seconds % 60

    return lpad(hours, 2, '0') * ":" *
           lpad(minutes, 2, '0') * ":" *
           lpad(seconds, 2, '0')
end


timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
log_dir = "logs/" * cfg_log["log_folder_prefix"] * "_log_$timestamp"
if !isdir(log_dir)
    mkpath(log_dir)
end

@info("log folder", log_dir=log_dir)

with_logger(FileLogger(joinpath(log_dir, "meta_data.log"))) do
    cfg_symbolized = Dict(Symbol(k) => v for (k, v) in cfg_sr)
    @info("Basic Information", thread_num=Threads.nthreads(), cfg_symbolized...)
end

#endregion

println("Imports Completed!")
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

function set_joint_data_x(x, info, d)
    joint_data =  zeros(length(x), cfg_data["num_dimensions"])
    joint_data[:, d] = x
    repeat_info = repeat(info', length(x))
    joint_data[:, 1:(d - 1)] = repeat_info[:, 1:(d - 1)]
    joint_data[:, (d + 1):cfg_data["num_dimensions"]] = repeat_info[:, d:end]
    return joint_data
end

joint_data_x = vcat([vcat([set_joint_data_x(x, info, d) for (x, info) in zip(c_xd[d], c_xd_slice_info[d])]...) for d in 1:cfg_data["num_dimensions"]]...)

joint_data_y = vcat([vcat([y*info for (y, info) in zip(c_yd[d], c_yd_slice_info[d])]...) for d in 1:cfg_data["num_dimensions"]]...)
#endregion

println("Data Loaded!")
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


#region Set options
# default_options = SymbolicRegression.Options(;
#     # Creating the Search Space
#     binary_operators=Function[+, -, /, *],
#     unary_operators=Function[],
#     maxsize=30,
#     # Setting the Search Size
#     populations=31,
#     population_size=27,
#     ncycles_per_iteration=380,
#     # Working with Complexities
#     parsimony=0.0,
#     warmup_maxsize_by=0.0,
#     adaptive_parsimony_scaling=1040,
#     # Mutations
#     mutation_weights=MutationWeights(;
#         mutate_constant=0.0346,
#         mutate_operator=0.293,
#         swap_operands=0.198,
#         rotate_tree=4.26,
#         add_node=2.47,
#         insert_node=0.0112,
#         delete_node=0.870,
#         simplify=0.00209,
#         randomize=0.000502,
#         do_nothing=0.273,
#         optimize=0.0,
#         form_connection=0.5,
#         break_connection=0.1,
#     ),
#     crossover_probability=0.0259,
#     annealing=true,
#     alpha=3.17,
#     perturbation_factor=0.129,
#     probability_negate_constant=0.00743,
#     # Tournament Selection
#     tournament_selection_n=15,
#     tournament_selection_p=0.982,
#     # Migration between Populations
#     fraction_replaced=0.00036,
#     ## ^Note: the optimal value found was 0.00000425,
#     ## but I thought this was a symptom of doing the sweep on such
#     ## a small problem, so I increased it to the older value of 0.00036
#     fraction_replaced_hof=0.0614,
#     topn=12,
#     # Performance and Parallelization
#     batching=false,
#     batch_size=50,
# )

options_marginal = []
for d in 1:cfg_data["num_dimensions"]
    global options_marginal
    options_marginal = [options_marginal; SymbolicRegression.Options(;
        binary_operators=cfg_sr["binary_operators"],
        unary_operators=cfg_sr["unary_operators"],
        constraints=cfg_sr["constraints"],
        nested_constraints=cfg_sr["nested_constraints"],
        output_directory=log_dir*"/marginal_$(d)",
        maxsize=cfg_sr["maxsize"],
        ncycles_per_iteration=cfg_sr["ncycles_per_iteration"],
        parsimony=cfg_sr["parsimony"],
        warmup_maxsize_by=cfg_sr["warmup_maxsize_by"],
        adaptive_parsimony_scaling=cfg_sr["adaptive_parsimony_scaling"],
        progress=cfg_sr["progress"],
        verbosity=cfg_sr["verbosity"]
    )]
end

d_slice_permutations = [(d, slice) for d in 1:cfg_data["num_dimensions"] for slice in 1:cfg_data["num_conditional_slices"]]
options_conditional = []
for (d, slice) in d_slice_permutations
    global options_conditional
    options_conditional = [options_conditional; SymbolicRegression.Options(;
        binary_operators=cfg_sr["binary_operators"],
        unary_operators=cfg_sr["unary_operators"],
        constraints=cfg_sr["constraints"],
        nested_constraints=cfg_sr["nested_constraints"],
        output_directory=log_dir*"/conditional_$(d)_$(slice)",
        maxsize=cfg_sr["maxsize"],
        ncycles_per_iteration=cfg_sr["ncycles_per_iteration"],
        parsimony=cfg_sr["parsimony"],
        warmup_maxsize_by=cfg_sr["warmup_maxsize_by"],
        adaptive_parsimony_scaling=cfg_sr["adaptive_parsimony_scaling"],
        progress=cfg_sr["progress"],
        verbosity=cfg_sr["verbosity"]
    )]
end


joint_options = SymbolicRegression.Options(;
    binary_operators=cfg_sr["binary_operators"], 
    unary_operators=cfg_sr["unary_operators"], 
    populations = cfg_sr["num_populations_for_joint_sr"], 
    population_size = cfg_sr["population_size_for_joint_sr"],
    
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
    populations = cfg_sr["num_populations_for_joint_sr"], 
    population_size = cfg_sr["population_size_for_joint_sr"],
    
    constraints=cfg_sr["constraints"],
    nested_constraints=cfg_sr["nested_constraints"],
    output_directory=log_dir*"/joint_no_init",
    maxsize=cfg_sr["maxsize"],
    ncycles_per_iteration=cfg_sr["ncycles_per_iteration"],
    parsimony=cfg_sr["parsimony"],
    warmup_maxsize_by=cfg_sr["warmup_maxsize_by"],
    adaptive_parsimony_scaling=cfg_sr["adaptive_parsimony_scaling"],
    progress=cfg_sr["progress"],
    verbosity=cfg_sr["verbosity"]
)
#endregion

println("Options Set!")

#region Marginal SR calls
marginal_halls_of_fame = Vector{Any}(undef, cfg_data["num_dimensions"])
dominating_pareto_marginals = Vector{Any}(undef, cfg_data["num_dimensions"])
trees_marginals = Vector{Any}(undef, cfg_data["num_dimensions"])
# loggers = [FileLogger(joinpath(log_dir, "marginal$(d).log")) for d in 1:cfg_data["num_dimensions"]]

@threads for d in 1:cfg_data["num_dimensions"]

    #region marginal_log
    # with_logger(loggers[d]) do
        # start_time = now()
        hall_of_fame = equation_search(
            reshape(m_xd[d], 1, :), m_yd[d]; 
            options=options_marginal[d], 
            parallelism=cfg_sr["parallelism_for_marginal_sr"], 
            niterations=cfg_sr["niterations_for_marginal_sr"]
        )
        # end_time = now()
        # duration = format_hms(end_time - start_time)
        # start_time = Dates.format(start_time, "yyyymmdd_HHMMSS")
        # end_time = Dates.format(end_time, "yyyymmdd_HHMMSS")
        # @info("Time Information", start_time=start_time, end_time=end_time, duration=duration)

        for member in eachindex(hall_of_fame.members)
            update_feature!(hall_of_fame.members[member].tree.tree, 1, d)
        end
        
        pareto = calculate_pareto_frontier(hall_of_fame)
        trees = [member.tree for member in pareto]

        marginal_halls_of_fame[d] = hall_of_fame
        dominating_pareto_marginals[d] = pareto
        trees_marginals[d] = trees
    # end
end
#endregion

println("Marginal SR Completed!")
# readline()


#region Conditional SR calls
conditional_halls_of_fame_per_slice = Vector{Any}(undef, cfg_data["num_conditional_slices"])
conditional_halls_of_fame = [conditional_halls_of_fame_per_slice for i in 1:cfg_data["num_dimensions"]]
dominating_pareto_conditionals_per_slice = Vector{Any}(undef, cfg_data["num_conditional_slices"])
dominating_pareto_conditionals = [dominating_pareto_conditionals_per_slice for i in 1:cfg_data["num_dimensions"]]
trees_conditionals_per_slice = Vector{Any}(undef, cfg_data["num_conditional_slices"])
trees_conditionals = [trees_conditionals_per_slice for i in 1:cfg_data["num_dimensions"]]

# loggers = Dict{Tuple{Int, Int}, AbstractLogger}()

# for (d, slice) in d_slice_permutations
#     log_path = joinpath(log_dir, "conditional_$(d)_$(slice).log")
#     loggers[(d, slice)] = FileLogger(log_path)
# end

@threads for iter in eachindex(d_slice_permutations)
    (d, slice) = d_slice_permutations[iter]

    x = reshape(c_xd[d][slice], 1, :)
    y = c_yd[d][slice]
    # with_logger(loggers[(d, slice)]) do
        # start_time = now()
        hall_of_fame = equation_search(
            x, y; 
            options=options_conditional[iter], 
            parallelism=cfg_sr["parallelism_for_conditional_sr"], 
            niterations=cfg_sr["niterations_for_conditional_sr"],
        )
        # end_time = now()
        # duration = format_hms(end_time - start_time)
        # start_time = Dates.format(start_time, "yyyymmdd_HHMMSS")
        # end_time = Dates.format(end_time, "yyyymmdd_HHMMSS")
        # @info("Time Information", start_time=start_time, end_time=end_time, duration=duration)

        for member in eachindex(hall_of_fame.members)
            update_feature!(hall_of_fame.members[member].tree.tree, 1, d)
        end

        pareto = calculate_pareto_frontier(hall_of_fame)
        trees = [member.tree for member in pareto]

        conditional_halls_of_fame[d][slice] = hall_of_fame
        dominating_pareto_conditionals[d][slice] = pareto
        trees_conditionals[d][slice] = trees
    # end
end
#endregion

println("Conditional SR Completed!")
# readline()

# Save the marginal and conditional hall of fame and pareto frontiers
save_path = joinpath(log_dir, "marginal_halls_of_fame.jls")
open(save_path, "w") do io
    serialize(io, marginal_halls_of_fame) # to load the data again use deserialze
end

save_path = joinpath(log_dir, "dominating_pareto_marginals.jls")
open(save_path, "w") do io
    serialize(io, dominating_pareto_marginals) # to load the data again use deserialze
end

save_path = joinpath(log_dir, "conditional_halls_of_fame.jls")
open(save_path, "w") do io
    serialize(io, conditional_halls_of_fame) # to load the data again use deserialze
end

save_path = joinpath(log_dir, "dominating_pareto_conditionals.jls")
open(save_path, "w") do io
    serialize(io, dominating_pareto_conditionals) # to load the data again use deserialze
end
