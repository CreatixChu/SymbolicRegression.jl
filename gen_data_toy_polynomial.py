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

def toy_2D_conditional_y(y, x_fix, y_high, y_low):
    marginal_prob = toy_2D_marginal_x(x_fix, y_high, y_low)
    x_fix = np.ones(np.shape(y))
    z = toy_2D_polynomial(y, x_fix)
    z = z/marginal_prob

x = np.linspace(0, 10, 100)
y = np.linspace(0, 10, 100)

# Generate data
y_low = 0
y_high = 10
x_low = 0
x_high = 10

# Generate marginals
marginal_x = toy_2D_marginal_x(x, y_high, y_low)
np.savetxt("toy_2D_polynomial_marginal_x.csv", marginal_x, delimiter=",")
marginal_y = toy_2D_marginal_y(y, x_high, x_low)
np.savetxt("toy_2D_polynomial_marginal_y.csv", marginal_y, delimiter=",")

# Generate conditionals
y_fix = [0.5, 2.2, 4.6, 5.8, 7.3, 8.9]
for i,y_fixed in enumerate(y_fix):
    conditional_x = toy_2D_conditional_x(x, y_fixed, x_high, x_low)
    np.savetxt(f"toy_2D_polynomial_conditional_x_{i}.csv", conditional_x, delimiter=",")
    conditional_y = toy_2D_conditional_y(y, y_fixed, y_high, y_low)
    np.savetxt(f"toy_2D_polynomial_conditional_y_{i}.csv", conditional_y, delimiter=",")
