using Pkg
Pkg.activate(".")
using SymbolicRegression
using CSV
using DataFrames
using Random
using Plots
using LoggingExtras
using TensorBoardLogger
using Dates
using Base.Threads
using FilePathsBase

timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
log_dir = "logs/log_$timestamp"
if !isdir(log_dir)
    mkpath(log_dir)
end

include("config.jl")
cfg=CONFIG

gr()

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

with_logger(FileLogger(joinpath(log_dir, "meta.txt"))) do
    cfg_symbolized = Dict(Symbol(k) => v for (k, v) in cfg)
    @info("Basic Information", thread_num=Threads.nthreads(), cfg_symbolized...)
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


# m_x1_p = plot(m_x[1], m_y[1], seriestype=:scatter, title="Scatter plot of data x0", xlabel="X-axis", ylabel="Y-axis")
# m_x2_p = plot(m_x[2], m_y[2], seriestype=:scatter, title="Scatter plot of data x1", xlabel="X-axis", ylabel="Y-axis")
# display(m_x1_p)
# display(m_x2_p)

#region Low level API
options = SymbolicRegression.Options(;
    binary_operators=cfg["binary_operators"], unary_operators=cfg["unary_operators"]
)

jobs = []
const hof_dict = Dict{Symbol, Any}()
const dominating_dict = Dict{Symbol, Any}()
const trees_dict = Dict{Symbol, Any}()

@info "preparing marginal jobs"
for i in 1:2
    var_name = Symbol("marginal_x$i")
    hof_dict[var_name] = nothing
    dominating_dict[var_name] = nothing
    trees_dict[var_name] = nothing
    push!(jobs, () -> begin
        @info "marginal$i start"
        with_logger(FileLogger(joinpath(log_dir, "marginal_x$i.txt"))) do
            hof_result = equation_search(
                reshape(m_x[i], 1, :), m_y[i]; 
                options=options,
                parallelism=cfg["parallelism_for_marginal_sr"],
                niterations=cfg["niterations_for_marginal_sr"],
                logger=SRLogger(current_logger(), log_interval=10)
            )
            hof_dict[var_name] = hof_result
            dominating_dict[var_name] = calculate_pareto_frontier(hof_result)
            trees_dict[var_name] = [member.tree for member in dominating_dict[var_name]]
            if i == 2
                for tree in trees_dict[var_name]
                    update_feature!(tree.tree, 1, 2)
                end
            end
        end
        @info "marginal$i finish"
    end)
end

@info "preparing conditional jobs"
for vid in 1:2
    data_x = getfield(Main, Symbol("conditional_data_x$vid"))
    data_y = getfield(Main, Symbol("conditional_data_y$vid"))
    n = length(data_x)

    for i in 1:n
        var_name = Symbol("conditional_x$(vid)_slice$i")
        hof_dict[var_name] = nothing
        dominating_dict[var_name] = nothing
        trees_dict[var_name] = nothing
        x = reshape(data_x[i][1], 1, :)
        y = data_y[i][1]
        push!(jobs, () -> begin
            @info "conditional$vid-slice$i start"
            with_logger(FileLogger(joinpath(log_dir, "conditional_x$(vid)_slice$i.txt"))) do
                hof_dict[var_name] = equation_search(
                    x, y; 
                    options=options, 
                    parallelism=cfg["parallelism_for_conditional_sr"], 
                    niterations=cfg["niterations_for_conditional_sr"],
                    logger=SRLogger(current_logger(), log_interval=10)
                )
                dominating_dict[var_name] = calculate_pareto_frontier(hof_dict[var_name])
                trees_dict[var_name] = [member.tree for member in dominating_dict[var_name]]
                if vid == 2
                    for tree in trees_dict[var_name]
                        update_feature!(tree.tree, 1, 2)
                    end
                end
            end
            @info "conditional$vid-slice$i finish"
        end)
    end
end

@info "start multithreading execution" jobs_num=length(jobs)
@threads for i in 1:length(jobs)
    jobs[i]()
end

@info "finsh multithreading execution!"
readline()

joint_initial_population = []

function multiply_conditionals_with_marginals(conditional_pop_members, marginal_pop_members)
    joint_pop_members = deepcopy(conditional_pop_members)
    for i in eachindex(joint_pop_members)
        joint_pop_members[i].tree = joint_pop_members[i].tree * rand(marginal_pop_members).tree
    end
    return joint_pop_members
end


for i in 1:8
    append!(joint_initial_population, multiply_conditionals_with_marginals(hof_dict[Symbol("conditional_x1_slice$i")][i].members, hof_dict["marginal_x2"].members))
end

for i in 1:8
    append!(joint_initial_population, multiply_conditionals_with_marginals(hof_dict[Symbol("conditional_x2_slice$i")][i].members, hof_dict["marginal_x1"].members))
end

shuffle(joint_initial_population)

populations = [joint_initial_population[i:i+29] for i in 1:30:480]

options1 = SymbolicRegression.Options(;
    binary_operators=cfg["binary_operators"], unary_operators=cfg["unary_operators"], populations = length(populations), population_size = length(populations[1])
    )

# println("Press any key to continue...at end")
# readline()

with_logger(FileLogger(joinpath(log_dir, "final.txt"))) do
    hof = equation_search(
        reshape(joint_data_x, 2, :), 
        joint_data_y; 
        options=options1, 
        parallelism=:serial, 
        initial_populations=populations, 
        niterations=cfg["niterations_for_joint_sr"],
        logger=SRLogger(current_logger(), log_interval=10)
    )
end







# Bug: something is wrong with the conditional slice probabilities in the dataset!!!!
# Multiply marginals and conditionals to obtain PopMember for initialization
# 30*8 = 240 conditionals
# 30 marginals
# 240*30 = 7200
# 7200*2 = 14,400
