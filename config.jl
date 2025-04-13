
pow2(x) = x^2
pow3(x) = x^3
pow4(x) = x^4
pow5(x) = x^5


const CONFIG = Dict(
    "data_path" => "./transfer",
    "num_dimensions" => 2,
    "num_conditional_slices" => 8,
    "max_kde_slice_dimension" => 1,
    "binary_operators" => [+, -, *, /],
    "unary_operators" => [exp, pow2, pow3, pow4, pow5],
    "parallelism_for_kde" => :serial,
    "parallelism_for_marginal_sr" => :multithreading,
    "niterations_for_marginal_sr" => 5,
    "parallelism_for_conditional_sr" => :multithreading,
    "niterations_for_conditional_sr" => 5,
    "parallelism_for_joint_sr" => :multithreading,
    "niterations_for_joint_sr" => 5,
    "num_populations_for_marginal_sr" => 15,
    "population_size_for_marginal_sr" => 30,
    "num_populations_for_conditional_sr" => 15,
    "population_size_for_conditional_sr" => 30,
    "num_populations_for_joint_sr" => 15,
    "population_size_for_joint_sr" => 30
)