using Symbolics
using LossFunctions

@variables x1, x2
expr = -7.4224068382544 * ((((0.41148044626858077 * (x1)^4) + ((-0.818348311927535 * x1) * (x2)^3)) + ((0.9289671840836182 * x1) * x2)) + (-0.6881469547522394 * x2))
simplified_expr = Symbolics.simplify(expr; expand=true)
println(simplified_expr)

true_f(x1, x2) = (1/1.075)*(x1^4 + x1*x2^3 + x1*x2 + x2)

# Should be the same as expr or simplified_expr
found_f(x1, x2) = 5.1077066626769625x2 - 6.895172379656182x1*x2 - 3.054175278191886(x1^4) + 6.07411410652488x1*(x2^3)

# Update the path to the latest hall of fame
joint_hall_of_fame = deserialize("/home/aa3rajen/projects/def-vganesh/aa3rajen/SymbolicRegression.jl/logs/cluster_log_20250514_200814/joint_hall_of_fame.jls")
joint_hall_of_fame.members[23].member

loss = L2DistLoss()

predict_true = true_f.(joint_data_x[:,1], joint_data_x[:,2])
predict_found = found_f.(joint_data_x[:,1], joint_data_x[:,2])
true_loss_val = LossFunctions.mean(loss, predict_true, joint_data_y)
found_loss_val = LossFunctions.mean(loss, predict_found, joint_data_y)
