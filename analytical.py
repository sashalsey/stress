import numpy as np
#### Analytical 1 #####
# # Material properties
# E = 1.0e11
# nu = 0.3
# alpha = 8e-6
# mu = E / (2 * (1 + nu))
# lmbda = E * nu / (1 - nu**2)

# epsilon_thermal = alpha * (0 - 100) * np.eye(2)
# print(epsilon_thermal)
# # [[-0.0008 -0.    ]
# #  [-0.     -0.0008]]

# # Mechanical strain (fully constrained)
# epsilon_mech = -epsilon_thermal

# sigma = lmbda * (epsilon_mech[0,0] + epsilon_mech[1,1]) * np.eye(2) + 2 * mu * epsilon_mech[:2,:2]
# print(sigma)
# # [[1.14285714e+08 0.00000000e+00]
# #  [0.00000000e+00 1.14285714e+08]]

#### Analytical 2 #####
alpha = 8e-6
n = 2
h = 0.05 / n
width = 0.01

def curvature_rho(T_0, T_1, T_2):
    return 1 / (1.5 * alpha * ((T_1 - T_0)- (T_2 - T_1))/ h )
def delta(rho): return width**2 / (8 * rho)
def theta(delta_): return np.rad2deg(np.arctan(delta_ / (width * 0.5)))

# a100_80_60 = theta(delta(curvature_rho(100, 80, 60)))
# print("100 - 80 - 60 theta {:.4f}".format(a100_80_60))

# a80_60_40 = theta(delta(curvature_rho(80, 60, 40)))
# print("80 - 60 - 40 theta {:.4f}".format(a80_60_40))

# a60_40_20 = theta(delta(curvature_rho(60, 40, 20)))
# print("60 - 40 - 20 theta {:.4f}".format(a60_40_20))

# a40_20_0 = theta(delta(curvature_rho(40, 20, 0)))
# print("40 - 20 - 0 theta {:.4f}".format(a40_20_0))

a20_0_0 = theta(delta(curvature_rho(1000, 0, 0)))
print("500 - 0 - 0 theta {:.4f}".format(a20_0_0))

T_1 = 0
T_2 = 1000
T_0 = 0
E = 1e11
stress = 0.5 * E * alpha * ((T_1 - T_0)- (T_2 - T_1))
print("stress {:.2f}".format(stress))
# # stress 80000000.00 = 8e7
