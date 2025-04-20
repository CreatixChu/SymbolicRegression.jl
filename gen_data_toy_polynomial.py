import numpy as np

def toy_2D_polynomial(x, y):
    z = x**4 + x*y**3 + y**2 + x*y + y + 3
    z = z*0.001
    return z

def toy_2D_marginal_x(x, y_high, y_low):
    func = lambda x, y : x**4*y + (x*y**4)/4 + (y**3)/3 + x*(y**2)/2 + (y**2)/2 + 3*y
    z = func(x, y_high) - func(x, y_low)
    z = z*0.01
    return z

def toy_2D_marginal_y(y, x_high, x_low):
    func = lambda x, y: (x**5)/5 + (x**2)*(y**3)/2 + x*(y**2) + (x**2)*y/2 + x*y + 3*x
    z = func(y, x_high) - func(y, x_low)
    z = z*0.01
    return z

def toy_2D_conditional_x(x, y_fix, x_high, x_low):
    marginal_prob = toy_2D_marginal_y(y_fix, x_high, x_low)
    y_fix = np.ones(np.shape(x))
    z = toy_2D_polynomial(x, y_fix)
    z = z/marginal_prob
    return z

def toy_2D_conditional_y(y, x_fix, y_high, y_low):
    marginal_prob = toy_2D_marginal_x(x_fix, y_high, y_low)
    x_fix = np.ones(np.shape(y))
    z = toy_2D_polynomial(y, x_fix)
    z = z/marginal_prob
    return z

x = np.linspace(0, 10, 100)
y = np.linspace(0, 10, 100)

# Generate data
y_low = 0
y_high = 10
x_low = 0
x_high = 10

# Generate marginals
marginal_x = toy_2D_marginal_x(x, y_high, y_low)
data = np.hstack([x.reshape(-1, 1), marginal_x.reshape(-1, 1)])
np.savetxt("./data/processed_data/toy_2D_polynomial_marginal_data_0.csv", data, delimiter=",")
marginal_y = toy_2D_marginal_y(y, x_high, x_low)
data = np.hstack([y.reshape(-1, 1), marginal_y.reshape(-1, 1)])
np.savetxt("./data/processed_data/toy_2D_polynomial_marginal_data_1.csv", data, delimiter=",")

# Generate conditionals
data_fix = np.random.choice(x, size=10, replace=False)
for i,data_fixed in enumerate(data_fix):
    conditional_x = toy_2D_conditional_x(x, data_fixed, x_high, x_low)
    data = np.hstack([x.reshape(-1, 1), conditional_x.reshape(-1, 1)])
    np.savetxt(f"./data/processed_data/toy_2D_polynomial_conditional_data_0_slice_{i}.csv", data, delimiter=",")
    conditional_y = toy_2D_conditional_y(y, data_fixed, y_high, y_low)
    data = np.hstack([y.reshape(-1, 1), conditional_y.reshape(-1, 1)])
    np.savetxt(f"./data/processed_data/toy_2D_polynomial_conditional_data_1_slice_{i}.csv", data, delimiter=",")

# Generate conditional slices
conditional_slices_x = np.zeros((len(data_fix), 2))
conditional_slices_y = np.zeros((len(data_fix), 2))
for i,data_fixed in enumerate(data_fix):
    conditional_slices_x[i][0] = data_fixed
    conditional_slices_x[i][1] = toy_2D_marginal_y(data_fixed, x_high, x_low)
    conditional_slices_y[i][0] = data_fixed
    conditional_slices_y[i][1] = toy_2D_marginal_y(data_fixed, y_high, y_low)
np.savetxt("./data/processed_data/toy_2D_polynomial_conditional_slices_0.csv", conditional_slices_x, delimiter=",")
np.savetxt("./data/processed_data/toy_2D_polynomial_conditional_slices_1.csv", conditional_slices_y, delimiter=",")
