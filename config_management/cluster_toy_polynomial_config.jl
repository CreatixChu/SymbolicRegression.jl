pow2(x) = x^2
pow3(x) = x^3
pow4(x) = x^4
pow5(x) = x^5

data_path = "./data/processed_data"
prefix = "toy_2D_polynomial"
files = readdir(data_path, join=true)
num_dimensions = count(f -> isfile(f) && startswith(basename(f), prefix * "_marginal_data"), files)
num_conditional_slices = count(f -> isfile(f) && startswith(basename(f), prefix * "_conditional_data_0_slice_"), files)

const CONFIG_data = Dict(
    "data_path_and_prefix" => data_path * "/" * prefix,
    "num_dimensions" => num_dimensions,
    "num_conditional_slices" => num_conditional_slices,
)

const CONFIG_log = Dict(
    "log_folder_prefix" => "cluster",
    "log_interval" => 1,
)

# operator/constant/variable all have complexity of 1
const CONFIG_sr = Dict(
    "binary_operators" => [+, -, *, /],
    "unary_operators" => [exp, log, pow2, pow3, pow4, pow5],
    "constraints" => [exp => 4, log => 4],
    "nested_constraints" => [exp => [exp => 0], log => [log => 0]],
    "maxsize" => 30,
    "ncycles_per_iteration" => 380,
    "parsimony" => 0.0,
    "warmup_maxsize_by" => 0.0,
    "adaptive_parsimony_scaling" => 1040,
    "parallelism_for_marginal_sr" => :multithreading,
    "parallelism_for_conditional_sr" => :multithreading,
    "parallelism_for_joint_sr" => :multithreading,
    "niterations_for_marginal_sr" => 4000,
    "niterations_for_conditional_sr" => 2000,
    "niterations_for_joint_sr" => 4000,
    "num_populations_for_marginal_sr" => 15,
    "num_populations_for_conditional_sr" => 15,
    "num_populations_for_joint_sr" => 15,
    "population_size_for_marginal_sr" => 30,
    "population_size_for_conditional_sr" => 30,
    "population_size_for_joint_sr" => 24,
    "joint_expression_possibilities" => "cartesian", # "one_to_one" or "cartesian"    
    "joint_max_num_expressions_per_dim_and_slice" => 30, # use Inf to avoid limiting growth in number of expressions per dim and slice when multiplying conditionals and marginals
    "joint_max_num_expressions" => 450 # use Inf to avoid limiting number of expressions
)
