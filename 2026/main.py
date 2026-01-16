# ===== 1) INITIALISE =====
import numpy as np
import firedrake as fd
import firedrake.adjoint as fda
from firedrake.output import VTKFile
from pyadjoint import get_working_tape
import cyipopt
import datetime
from pathlib import Path
import os
basename = os.path.splitext(os.path.basename(__file__))[0]
outputFolder = (f"results/{basename}_" + datetime.datetime.now().strftime("%d-%H-%M-%S") + "/")

tape = get_working_tape()
tape.clear_tape()
fda.continue_annotation()

##### geometry/mesh #####
nx, ny = 20, 40
lx, ly = 0.05, 0.1
cellsize, xcellsize = ly / ny, lx / nx
mesh = fd.RectangleMesh(nx, ny, lx, ly, quadrilateral=True)
meshVolume = fd.assemble(1 * fd.dx(domain=mesh))
Id = fd.Identity(mesh.geometric_dimension())
helmholtzFilterRadius = 1 * (meshVolume / mesh.num_cells()) ** (1 / mesh.cell_dimension())

##### function spaces #####
densityspace = fd.FunctionSpace(mesh, "CG", 2)
displace = fd.VectorFunctionSpace(mesh, "CG", 2)

##### functions anf files #####
rho = fd.Function(densityspace)
rho_hatFunction = fd.Function(densityspace, name="rho_hat")

uFunction = fd.Function(displace, name="u")     # displacement due to in-service load

numberOfNodes = len(rho.vector().get_local())
numberOfVariables = rho.dat.data_ro.size
lowerBounds = np.zeros(numberOfVariables)
upperBounds = np.ones(numberOfVariables)

##### design variable #####
Vlimit = 0.5
rho.assign(Vlimit)

##### material properties #####
E_0, E_1, penalisationExponent = 1.0e2, 1.19e11, 3.0             # Young’s moduli and penalisation
nu = 0.3                                                        # Poisson’s ratio

##### hyperbolic tangent parameters #####
beta0, eta0 = 8.0, 0.5

##### loads and BCs #####
# 1: vertical y lhs, 2: vertical y rhs, 3: horizontal x bottom, 4: horizontal x top
mech_bc = fd.DirichletBC(displace, fd.Constant((0.0, 0.0)), 3)

##################################################
# ===== 2) DEF OBJ / GRAD (In = numpy array) =====

def obj_grad(In: np.ndarray):
    fda.continue_annotation()

    rho_control = fd.Function(densityspace)
    rho_ = fd.Function(densityspace)
    sol_u = fd.Function(displace)

    rho_control.dat.data[:] = In

    ##### Helmholtz Filter #####
    u_h = fd.TrialFunction(densityspace)
    v_h = fd.TestFunction(densityspace)
    a_h = (fd.Constant(helmholtzFilterRadius) ** 2) * (fd.dot(fd.grad(u_h), fd.grad(v_h)) * fd.dx) + fd.inner(u_h, v_h) * fd.dx
    L_h = fd.inner(rho_control, v_h) * fd.dx
    fd.solve(a_h == L_h, rho_)

    #### Projection Filter #####
    rho_hat = (fd.tanh(beta0 * eta0) + fd.tanh(beta0 * (rho_ - eta0))) / (fd.tanh(beta0 * eta0) + fd.tanh(beta0 * (1 - eta0)))
    rho_hatFunction.assign(fd.project(rho_hat, densityspace))

    ##### Compliance: Linear Elasticity #####
    x, y = fd.SpatialCoordinate(mesh)
    x_point, y_point = lx - 1*xcellsize, ly - 2*cellsize
    area = 2*cellsize
    Force = fd.conditional(fd.And(fd.gt(x, x_point),fd.gt(y, y_point)) ,fd.as_vector([1e6/area, 0]),fd.as_vector([0, 0]))

    E = E_0 + (E_1 - E_0) * (rho_hat ** penalisationExponent)
    u_c = fd.TrialFunction(displace)
    v_c = fd.TestFunction(displace)
    
    lambda_ = (E * nu) / ((1 + nu) * (1 - 2 * nu))
    mu = (E) / (2 * (1 + nu))
    epsilon = 0.5 * (fd.grad(v_c) + fd.grad(v_c).T)
    sigma = (lambda_ * fd.div(u_c) * Id) + (2 * mu * 0.5 * (fd.grad(u_c) + fd.grad(u_c).T))

    a_c = fd.inner(sigma, epsilon) * fd.dx
    L_c = fd.dot(Force, v_c) * fd.ds(2)
    
    fd.solve(a_c == L_c, sol_u, bcs=mech_bc)
    uFunction.assign(sol_u)

    ##### Objective #####
    j_adj = fd.assemble(fd.inner(Force, sol_u) * fd.ds(2))
    
    ##### Constraints #####
    vol_adj = (1.0 / meshVolume) * fd.assemble(rho_hat * fd.dx)

    djdrho = (fda.compute_gradient(j_adj,   fda.Control(rho_control)).riesz_representation('L2').vector().get_local())
    dc1drho = (fda.compute_gradient(vol_adj, fda.Control(rho_control)).riesz_representation('L2').vector().get_local())

    c = np.array([vol_adj])
    dcdrho_flat = dc1drho

    # tape.clear_tape()
    return j_adj, djdrho, c, dcdrho_flat

#################################################
# ===== 3) FINITE DIFFERENCE CHECK =====
x0 = rho.dat.data_ro.copy()
j0, djdrho0, c0, dcdrho0 = obj_grad(x0)
def finite_difference_check(perturbation_index, h):
    x1 = x0.copy()
    x1[perturbation_index] += h

    j1, _, c1, _ = obj_grad(x1)

    # Objective
    fd_obj = (j1 - j0) / h
    ad_obj = djdrho0[perturbation_index]
    obj_err = 100.0 * (fd_obj - ad_obj) / fd_obj

    # Constraints
    con_errs = []
    for i in range(len(c0)):
        fd_con = (c1[i] - c0[i]) / h
        ad_con = dcdrho0[i * numberOfVariables + perturbation_index]
        con_errs.append(100.0 * (fd_con - ad_con) / fd_con)

    return obj_err, np.array(con_errs)

obj_error_fn = fd.Function(densityspace, name="FD_Objective_Err")
con_error_fns = [fd.Function(densityspace, name=f"FD_Constraint_{i+1}_Err") for i in range(len(c0))]

h = 1e-4

obj_err_array = np.zeros(numberOfNodes)
con_err_arrays = [np.zeros(numberOfNodes) for _ in range(len(c0))]

for i in range(numberOfNodes):
    obj_err, con_errs = finite_difference_check(i, h)

    obj_err_array[i] = obj_err
    for k in range(len(c0)):
        con_err_arrays[k][i] = con_errs[k]
obj_error_fn.dat.data[:] = obj_err_array

for k in range(len(c0)):
    con_error_fns[k].dat.data[:] = con_err_arrays[k]

vtk = fd.VTKFile(outputFolder + f"finite_difference_errs.pvd")
vtk.write(obj_error_fn, *con_error_fns)
