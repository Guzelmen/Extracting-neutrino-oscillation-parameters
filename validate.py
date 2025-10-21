import numpy as np
from minimisers import *

params = [200, data, pred]

print("1D minimiser: parabolic method")
print("Function being tested: f(x) = sin(x)**2 + 0.1*x**2")
print("Analytic minimum of the function occurs at x = 0")
min_para, f_para, ity = parabolic_method(test_1d, -1, 1, [2], 0, params, 1e-8)
print(f"Value of x at the minimum using method: {min_para}")

print("\n2D tester:")
print("Function being tested: f(x, y) = (1 - x)**2 + 100 * (y - x**2)**2")
print("Analytic minimum of the function occurs at (1, 1)")
x_univ, y_univ, it = univariate_method(rosen_2d, [[0.5, 1.5], [0.5, 1.5]], 1e-6)
print("\nUnivariate method:")
print(f"Minimum at (x, y) = ({round(x_univ[-1], 1)}, {round(y_univ[-1], 1)})")

min_grad = gradient_method(rosen_2d, [0.5, 0.5], params)
print("\nGradient method:")
print(f"Minimum at (x, y) = ({round(min_grad[0], 1)}, {round(min_grad[1], 1)})")

min_newt = newtons_method(rosen_2d, [0.5, 0.5], params)
print("\nNewton method:")
print(f"Minimum at (x, y) = ({round(min_newt[0], 1)}, {round(min_newt[1], 1)})")

min_quas = quasi_newton_method(rosen_2d, [0.5, 0.5], params, update_method = "Broyden")
print("\nQuasi-Newton method with Broyden as update:")
#print(f"Minimum at (x, y) = ({round(min_quas[0], 1)}, {round(min_quas[1], 1)})")