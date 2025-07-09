import numpy as np

# Material properties
E = 1.0e11
nu = 0.3
alpha = 8e-6
mu = E / (2 * (1 + nu))
lmbda = E * nu / (1 - nu**2)

epsilon_thermal = alpha * (0 - 100) * np.eye(2)
print(epsilon_thermal)
# [[-0.0008 -0.    ]
#  [-0.     -0.0008]]

# Mechanical strain (fully constrained)
epsilon_mech = -epsilon_thermal

sigma = lmbda * (epsilon_mech[0,0] + epsilon_mech[1,1]) * np.eye(2) + 2 * mu * epsilon_mech[:2,:2]
print(sigma)
# [[1.14285714e+08 0.00000000e+00]
#  [0.00000000e+00 1.14285714e+08]]

