n = 9
ly = 0.1
substrate_height = 0.01
layer_height = (ly - substrate_height) / n
print("layer height ", layer_height)
k = 100

for i in range(n):
    print("i = ", i)
    layer_height_dep = layer_height * (i + 1) + substrate_height
    print("Layer height dep ", layer_height_dep)
    print("layer_height_dep - layer_height ", layer_height_dep - layer_height)


      # x_min, x_max = 0.0, 0.05
            # y_min, y_max = 0.0, 0.1
            # tol = 0.01
            # slope1 = (y_max - y_min) / (x_max - x_min)
            # diag1_eq = y_min + slope1 * (x - x_min)  # Equation: y = slope1 * (x - x_min) + y_min
            # diag1 = fd.And(
            #     fd.ge(y, diag1_eq - tol),
            #     fd.le(y, diag1_eq + tol)
            # )

            # # Diagonal 2: From (x_min, y_max) to (x_max, y_min)
            # slope2 = (y_min - y_max) / (x_max - x_min)
            # diag2_eq = y_max + slope2 * (x - x_min)  # Equation: y = slope2 * (x - x_min) + y_max
            # diag2 = fd.And(
            #     fd.ge(y, diag2_eq - tol),
            #     fd.le(y, diag2_eq + tol)
            # )

            # # Combine diagonals (logical OR)
            # X_shape = fd.conditional(
            #     fd.Or(diag1, diag2),
            #     fd.Constant(1.0),  # Inside X
            #     fd.Constant(0.0)   # Outside X
            # )
            
            # self.rho_.interpolate(
            #     fd.conditional(fd.lt(x,0.02),
            #     fd.Constant(1.0),
            #     fd.Constant(0.0))
            #     +
            #     fd.conditional(fd.gt(x,0.03),
            #     fd.Constant(1.0),
            #     fd.Constant(0.0))
            #     +
            #     fd.conditional(fd.And(fd.ge(x,0.02),fd.And(fd.le(x,0.03),fd.lt(y,0.02))),
            #     fd.Constant(1.0),
            #     fd.Constant(0.0))
            #     +
            #     fd.conditional(fd.And(fd.ge(x,0.02),fd.And(fd.le(x,0.03),fd.gt(y,0.08))),
            #     fd.Constant(1.0),
            #     fd.Constant(0.0)))
            # self.rho_.interpolate(fd.conditional(fd.And(fd.ge(x,0.01),
            #     fd.le(x,0.04)),
            #     fd.Constant(1.0),
            #     fd.Constant(0.0)))
            angle_degrees = 90  # Set desired angle (0° to 90°)
            x_min, x_max = 0.0, self.lx
            y_min, y_max = 0.0, self.ly
            width = 0.025  # Total width of the line (constant)

            x_center = (x_min + x_max) / 2
            y_center = (y_min + y_max) / 2

            if angle_degrees == 90:
                line_lower = x_center - width / 2
                line_upper = x_center + width / 2
                line_cond = fd.And(
                    fd.ge(x, line_lower),
                    fd.le(x, line_upper))
            else:
                angle_rad = np.deg2rad(angle_degrees)
                slope = np.tan(angle_rad)
                intercept = y_center - slope * x_center
                
                offset = (width / 2) * np.sqrt(1 + slope**2)
                
                upper_intercept = intercept + offset
                lower_intercept = intercept - offset
                
                line_cond = fd.And(
                    fd.ge(y, slope * x + lower_intercept),
                    fd.le(y, slope * x + upper_intercept))

            rotatable_line = fd.conditional(line_cond,
                fd.Constant(1.0),
                fd.Constant(0.0))