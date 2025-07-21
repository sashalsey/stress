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
width = 0.1
T_0 = 100
T_1 = 80
T_2 = 0
curvature_rho = 1/(1.5 * alpha * ((T_1 - T_0)- (T_2 - T_0))/ h )
delta = width**2 / (8 * curvature_rho)
theta = np.rad2deg(np.arctan(delta / (width * 0.5)))

print("100 - 80 - 0 theta {:.4f}".format(theta))
# theta 0.14

T_0 = 100
T_1 = 60
T_2 = 0
curvature_rho = 1/(1.5 * alpha * ((T_1 - T_0)- (T_2 - T_0))/ h )
delta = width**2 / (8 * curvature_rho)
theta = np.rad2deg(np.arctan(delta / (width * 0.5)))

print("100 - 60 - 0 theta {:.4f}".format(theta))

T_0 = 100
T_1 = 20
T_2 = 0
curvature_rho = 1/(1.5 * alpha * ((T_1 - T_0)- (T_2 - T_0))/ h )
delta = width**2 / (8 * curvature_rho)
theta = np.rad2deg(np.arctan(delta / (width * 0.5)))

print("100 - 20 - 0 theta {:.4f}".format(theta))

# L_x = width
# L_y = 0.05
# u_x = -1.7e-4
# u_y = -1.7e-4
# gamma = np.rad2deg(np.arctan(u_y/L_x)+np.arctan(u_x/L_y))

# print("gamma {:.2f}".format(gamma))

# E = 1e11
# stress = 0.5 * E * alpha * ((T_1 - T_0)- (T_2 - T_0))
# print("stress {:.2f}".format(stress))
# # stress 80000000.00 = 8e7
