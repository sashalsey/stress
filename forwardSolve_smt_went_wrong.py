import numpy as np  # type: ignore
import firedrake as fd  # type: ignore
import firedrake.adjoint as fda  # type: ignore
from firedrake.output import VTKFile # type: ignore
import matplotlib.pyplot as plt # type: ignore
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type:ignore
from fix import FixedDirichletBC
import time
import sys
from firedrake import LinearVariationalProblem, LinearVariationalSolver

fda.continue_annotation()

class ForwardSolve:
    def __init__(self, outputFolder, outputFolder2, beta, penalisationExponent, variableInitialisation, rho0, pnorm, scenario, initial_stressintegral):
        # append inputs to class
        self.outputFolder = outputFolder
        self.outputFolder2 = outputFolder2

        # rho initialisation
        self.scenario = scenario
        self.variableInitialisation = variableInitialisation
        self.penalisationExponent = penalisationExponent
        self.rho0 = rho0

        # material properties
        self.E_0, self.E_1 = 0.01, 1.0e11
        self.nu = 0.3

        # thermal properties
        self.T_0, self.T_1 = 300.0, 2000.0 # atmospheric temperature, heating temperature
        self.k_0, self.k_1 = 0.026, 17.0 # thermal conduction coeff of air, titanium
        self.alpha_0, self.alpha_1 = 1e-12, 8e-6 # linear thermal expansion coeff of air, titanium
        self.h = 100.0 # thermal convection coeff
        self.cp_0, self.cp_1 = 1e3, 500 # coeff specific heat capacity of air, titanium
        self.rho_material = 4500.0
        self.t_heat = 5.0 # time between deposit and next layer
        self.dt = 2.0 # timestep for t_heat
        
        # pseudo-density functional
        self.beta = beta  # heaviside projection paramesteepnesster
        self.eta0 = 0.5  # midpoint of projection filter
        
        self.Vlimit = 0.4 # volume fraction constraint
        self.pnorm = pnorm # p-norm stress penalty
        self.initial_stressintegral = float(initial_stressintegral)

    def GenerateMesh(self,):
        self.mesh = fd.RectangleMesh(self.nx, self.ny, self.lx, self.ly, quadrilateral=True)
        self.gradientScale = (self.nx * self.ny) / (self.lx * self.ly)
        self.dx = fd.Measure('dx', domain=self.mesh)
        self.ds = fd.ds(domain=self.mesh)
        self.n = fd.FacetNormal(self.mesh)

    def Setup(self):
        self.nx, self.ny = 75, 150
        self.lx, self.ly =  0.05, 0.1
        self.cellsize = self.lx / self.nx
        self.GenerateMesh()
        self.meshVolume = fd.assemble(1 * fd.dx(domain=self.mesh))
                                   
        # function spaces
        self.densityspace = fd.FunctionSpace(self.mesh, "CG", 1)
        self.displace = fd.VectorFunctionSpace(self.mesh, "CG", 1)
        self.templace = fd.FunctionSpace(self.mesh, "CG", 1)
        self.DG0 = fd.TensorFunctionSpace(self.mesh, "DG", 0)

        self.rho = fd.Function(self.densityspace)

        # number of nodes in mesh
        self.numberOfNodes = len(self.rho.vector().get_local())

        # compute helmholtz filter radius
        self.helmholtzFilterRadius = 1 * (self.meshVolume / self.mesh.num_cells()) ** (1 / self.mesh.cell_dimension())

        # boundary conditions
        # 1: vertical y lhs, 2: vertical y rhs, 3: horizontal x bottom, 4: horizontal x top
        self.mech_bc = fd.DirichletBC(self.displace, fd.Constant((0.0, 0.0)), 3)
        self.mech_bc2 = FixedDirichletBC(self.displace, fd.Constant((0.0, 0.0)), np.array([0.026, 0.0]))
        self.mech_bc3 = FixedDirichletBC(self.displace, fd.Constant((0.0, 0.0)), np.array([0.024, 0.0]))
        self.temp_bc1 = fd.DirichletBC(self.templace, fd.Constant(self.T_0), 3)

        # output files
        self.rho_hatFunction =fd.Function(self.densityspace, name="rho_hat")
        self.rho_hatFile = VTKFile(self.outputFolder + "rho_hat.pvd")

        self.rho_dep =fd.Function(self.densityspace, name="rho_dep") # deposited rho 
        self.E = fd.Function(self.densityspace, name="E")
        self.k_rho = fd.Function(self.densityspace, name="k_rho") # density-dependent thermal conductivity
        self.cp = fd.Function(self.densityspace, name="cp") # density-dependent thermal storage
        self.alpha = fd.Function(self.densityspace, name="alpha")
        
        # self.alphaFile = VTKFile(self.outputFolder + "alpha.pvd")
        # self.cpFile = VTKFile(self.outputFolder + "cp.pvd")
        # self.EFile = VTKFile(self.outputFolder + "E.pvd")
        # self.k_rhoFile = VTKFile(self.outputFolder + "k_rho.pvd")
        self.rho_depFile = VTKFile(self.outputFolder + "rho_dep.pvd")

        self.tempFile = VTKFile(self.outputFolder + "temp.pvd")

        self.T_n = fd.Function(self.templace, name="Temperature")
        self.T_n1 = fd.Function(self.templace)
        
        self.strain_therm_increment = fd.Function(self.DG0, name="strain_therm_increment")
        self.strain_therm = fd.Function(self.DG0, name="strain_therm")
        self.strain_total_increment = fd.Function(self.DG0, name="strain_total_increment")
        self.strain_total = fd.Function(self.DG0, name="strain_total")
        self.strain_mech_increment = fd.Function(self.DG0, name="strain_mech_increment")
        self.strain_mech = fd.Function(self.DG0, name="strain_mech")
        self.stress_mech = fd.Function(self.DG0, name="stress_mech")

        self.strain_therm_incrementFile = VTKFile(self.outputFolder + "strain_therm_increment.pvd")
        self.strain_thermFile = VTKFile(self.outputFolder + "strain_therm.pvd")    
        self.strain_total_incrementFile = VTKFile(self.outputFolder + "strain_total_increment.pvd")
        self.strain_totalFile = VTKFile(self.outputFolder + "strain_total.pvd")
        self.strain_mech_incrementFile = VTKFile(self.outputFolder + "strain_mech_increment.pvd")
        self.strain_mechFile = VTKFile(self.outputFolder + "strain_mech.pvd")
        self.stress_mechFile = VTKFile(self.outputFolder + "stress_mech.pvd")

        # self.mech_stress_sensitivityFile = VTKFile(self.outputFolder + "mech_stress_sensitivity.pvd")
        # self.mech_stress_sensitivityFun = fd.Function(self.DG0)

        # self.von_mises_stress = fd.Function(self.densityspace, name="von")
        # self.von_mises_stressFile = VTKFile(self.outputFolder + "von.pvd")
    
        # mechanical distortion displacement
        self.uFile = VTKFile(self.outputFolder + "u.pvd")
        self.uFunction = fd.Function(self.displace, name="u")

        # thermal distortion displacement
        self.u_tFile = VTKFile(self.outputFolder + "u_t.pvd")
        self.u_tFunction = fd.Function(self.displace, name="u_t")

        self.j_tFile = VTKFile(self.outputFolder + "j_t.pvd")

    def ComputeInitialSolution(self):

        if self.variableInitialisation is True:
            initialisationFunction =fd.Function(self.densityspace)
            initialisationFunction.vector().set_local(self.rho0)
            print("Initialisation with previous solution")
        else:
            # initialise function for initial solution
            initialisationFunction =fd.Function(self.densityspace)

            # generate uniform initialisation
            initialisationFunction.assign(self.Vlimit)
            print("Initialisation: ", self.Vlimit)

        return initialisationFunction.vector().get_local()

    def CacheDesignVariables(self, designVariables, initialise=False):
        if initialise is True:
            # initialise with zero length array
            self.rho_np_previous = np.zeros(0)
            cache = False
        else:
            # determine whether the current design variables were simulated at previous iteration
            if np.array_equal(designVariables, self.rho_np_previous):
                # update cache boolean
                cache = True
            else:
                # new array is unique, assign current array to cache
                self.rho_np_previous = designVariables

                # update self.rho
                self.rho.vector().set_local(designVariables)

                # update cache boolean
                cache = False
        return cache

    def Solve(self, designVariables):
        # insert and cache design variables, automatically updates self.rho
        identicalVariables = self.CacheDesignVariables(designVariables)

        if identicalVariables is False:
            start_time = time.time()
            # Helmholtz Filter
            u = fd.TrialFunction(self.densityspace)
            v = fd.TestFunction(self.densityspace)

            x, y = fd.SpatialCoordinate(self.mesh)

            # Weak variational form
            a = (fd.Constant(self.helmholtzFilterRadius) ** 2) * fd.inner(fd.grad(u), fd.grad(v)) * self.dx + fd.inner(u, v) * self.dx
            L = fd.inner(self.rho, v) * self.dx

            # Solve Helmholtz equation
            self.rho_ = fd.Function(self.densityspace, name="rho_")
            fd.solve(a == L, self.rho_)

            # Projection filter
            self.rho_hat = (fd.tanh(self.beta * self.eta0) + fd.tanh(self.beta * (self.rho_ - self.eta0))
            ) / (fd.tanh(self.beta * self.eta0) + fd.tanh(self.beta * (1 - self.eta0)))

            self.rho_hatFunction.assign(fd.project(self.rho_hat, self.densityspace))
            self.rho_hatFile.write(self.rho_hatFunction)
            
            ##### Temperature ######
            # Total strain = strain_mech + strain_therm
            def total_strain_def(u): return 0.5 * (fd.grad(u) + fd.grad(u).T)

            k = 500
            # Initialise functions

            def thermal_strain_solve(self, T_n1, T_n):
                thermal_strain = self.alpha * (T_n1 - T_n) * self.rho_dep * Id
                self.strain_therm_increment.interpolate(fd.project(thermal_strain, self.DG0))
                self.strain_therm_increment.interpolate(self.strain_therm_increment * (0.5 * (1 + fd.tanh(k * (self.rho_dep - 0.5)))))
                # self.strain_therm_incrementFile.write(self.strain_therm_increment)

                self.strain_therm.interpolate(self.strain_therm_increment + self.strain_therm)
                self.strain_thermFile.write(self.strain_therm)
            
            distortion_solver_parameters = {
                'ksp_type': 'preonly',        # Use direct solver (recommended for small problems)
                'pc_type': 'lu',              # LU factorization
                'pc_factor_mat_solver_type': 'mumps',  # Fast direct solver
                'mat_mumps_icntl_24': 1,      # Enable error analysis in MUMPS
                'ksp_rtol': 1e-12,            # Reduced relative tolerance
                'ksp_atol': 1e-12,            # Absolute tolerance
                'ksp_max_it': 500,           # Maximum iterations
                'snes_rtol': 1e-12,           # Nonlinear solver tolerances
                'snes_atol': 1e-12,
                'snes_max_it': 500}

            def distortion_solve(self):
                u_t = fd.TrialFunction(self.displace)
                v_t = fd.TestFunction(self.displace)

                #####INCREMENTAL#####
                a_t = fd.inner((self.E * total_strain_def(u_t)), total_strain_def(v_t)) * self.rho_dep * fd.dx
                L_t = fd.inner((self.E * self.strain_therm_increment), total_strain_def(v_t)) * self.rho_dep * fd.dx
                sol = fd.Function(self.displace)
                fd.solve(a_t == L_t, sol, bcs=[self.mech_bc2, self.mech_bc3], solver_parameters=distortion_solver_parameters)
                
                self.u_tFunction.interpolate(self.u_tFunction + sol)
                # print("Type of self.u_tFunction:", type(self.u_tFunction))
                # print("Shape of self.u_tFunction:", self.u_tFunction.ufl_shape)

                self.strain_total_increment.interpolate(fd.project(total_strain_def(sol) * self.rho_dep, self.DG0))
                self.strain_total.interpolate(self.strain_total + self.strain_total_increment)
                self.strain_mech_increment.interpolate(self.strain_total_increment - self.strain_therm_increment)
                self.strain_mech.interpolate(self.strain_mech + self.strain_mech_increment)

                # print("Type of self.strain_total_increment:", type(self.strain_total_increment))
                # print("Shape of self.strain_total_increment:", self.strain_total_increment.ufl_shape)
                # print("Type of self.strain_mech:", type(self.strain_mech))
                # print("Shape of self.strain_mech:", self.strain_mech.ufl_shape)

                self.u_tFile.write(self.u_tFunction)
                self.strain_mechFile.write(self.strain_mech)
                self.stress_mech.interpolate(self.E * self.strain_mech)
                self.stress_mechFile.write(self.stress_mech)
                # print("Type of self.stress_mech:", type(self.stress_mech))
                # print("Shape of self.stress_mech:", self.stress_mech.ufl_shape)
 
            Id = fd.Identity(self.mesh.geometric_dimension())
            current_time = 0.0

            self.T_n.interpolate(fd.Constant(self.T_0))
            self.tempFile.write(self.T_n)
            self.T_n1.interpolate(fd.Constant(self.T_0))

            n = 10
            layer_height = self.ly / n
            # print("Layer height = ", layer_height)
            current_global_time = 0.0

            for i in range(n):
                print("i = ", i)
                layer_height_dep = layer_height * (i + 1)
                # rho_dep is 1 where rho_hat > 0.5, and 0 where rho_hat < 0.5, this avoids intermediate densities, especially during initialisation
                self.rho_dep.interpolate(0.5 * (1 + fd.tanh(k * ( - y + layer_height_dep))) * (0.5 * (1 + fd.tanh(k * (self.rho_hat - 0.5*self.Vlimit)))))
                self.k_rho.interpolate(self.k_0 + (self.k_1 - self.k_0) * (self.rho_dep ** self.penalisationExponent))
                self.alpha.interpolate(self.alpha_0 + (self.alpha_1 - self.alpha_0) * (self.rho_dep ** self.penalisationExponent))
                self.cp.interpolate(1.2 * self.cp_0 + (self.rho_material * self.cp_1 - 1.2 * self.cp_0) * (self.rho_dep ** self.penalisationExponent))
                self.T_n.interpolate(self.T_n * (1 + 0.5 * (fd.tanh(k * ( - y + layer_height_dep -layer_height)) + fd.tanh(k * (y - layer_height_dep))) * 0.5 * (1 + fd.tanh(k * (self.rho_dep - 0.5)))) 
                                    + self.T_1 * (-0.5*(fd.tanh(k*(-y+layer_height_dep-layer_height))+fd.tanh(k*(y-layer_height_dep))))*(0.5*(1+fd.tanh(k*(self.rho_dep - 0.5)))))
                self.E.interpolate(self.E_0 + (self.E_1 - self.E_0) * (self.rho_dep ** self.penalisationExponent))
                self.rho_depFile.write(self.rho_dep)

                self.tempFile.write(self.T_n)
                print("Max temp {:.2f}".format(np.max(self.T_n.vector().get_local())))
                current_time = 0.0

                while current_time < self.t_heat:
                    #### Temperature Solve ####
                    w = fd.TestFunction(self.templace)
                    Trial_n1 = fd.TrialFunction(self.templace)
                    F = (self.cp * (Trial_n1 - self.T_n) * w * self.dx + self.dt * self.k_rho * fd.dot(fd.grad(Trial_n1), fd.grad(w)) * self.dx) #+ (self.dt * self.h * (Trial_n1 - self.T_0) * w) * (self.ds(1) + self.ds(2) + self.ds(4))
                    F += self.dt * self.h * (Trial_n1 - self.T_0) * w * self.ds(1)
                    F += self.dt * self.h * (Trial_n1 - self.T_0) * w * self.ds(2)
                    F += self.dt * self.h * (Trial_n1 - self.T_0) * w * self.ds(4)
                    sol = fd.Function(self.templace)
                    a = fd.lhs(F)
                    L = fd.rhs(F)
                    fd.solve(a == L, sol, bcs=[self.temp_bc1])
                    self.T_n1.interpolate(sol)
                    
                    #### Distortion Solve ####
                    # Here, new temp is T_n1, previous temp is T_n
                    thermal_strain_solve(self, self.T_n1, self.T_n)
                    distortion_solve(self)

                    self.T_n.assign(self.T_n1)
                    self.tempFile.write(self.T_n)

                    print("Max temp {:.2f}".format(np.max(self.T_n.vector().get_local())))

                    current_time += self.dt
            #         current_global_time += self.dt
            #         time_steps.append(current_global_time)
            #         store_stuff(self)

            #         if i == n - 1: # Forced cooled to ambient
            #             temp_tolerance = 5.0  # Temperature difference considered "cooled"
            # #             max_iterations = 500   # Safety limit to prevent infinite loops
            #             iteration = 0
            #             while True:
            #                 w = fd.TestFunction(self.templace)
            #                 Trial_n1 = fd.TrialFunction(self.templace)
            #                 F = (self.cp * (Trial_n1 - self.T_n) * w * self.dx + self.dt * self.k_rho * fd.dot(fd.grad(Trial_n1), fd.grad(w)) * self.dx) #+ (self.dt * self.h * (Trial_n1 - self.T_0) * w) * (self.ds(1) + self.ds(2) + self.ds(4))
            #                 F += self.dt * self.h * (Trial_n1 - self.T_0) * w * self.ds(1)
            #                 F += self.dt * self.h * (Trial_n1 - self.T_0) * w * self.ds(2)
            #                 F += self.dt * self.h * (Trial_n1 - self.T_0) * w * self.ds(4)
            #                 sol = fd.Function(self.templace)
            #                 a = fd.lhs(F)
            #                 L = fd.rhs(F)
            #                 fd.solve(a == L, sol, bcs=[self.temp_bc1])
            #                 self.T_n1.interpolate(sol)
                            
            #                 # Here, new temp is T_n1, previous temp is T_n
            #                 thermal_strain_solve(self, self.T_n1, self.T_n)
            #                 distortion_solve(self)

            #                 self.T_n.assign(self.T_n1)
            #                 self.tempFile.write(self.T_n)

            #                 print("Max temp {:.2f}".format(np.max(self.T_n.vector().get_local())))

            #                 current_time += self.dt_2
            # #                 current_global_time += self.dt_2
            # #                 time_steps.append(current_global_time)
            # #                 store_stuff(self)

            #                 temp_diff = np.max(self.T_n.vector().get_local()) - self.T_0
                            
            # #                 # Temp convergence
            #                 if temp_diff < temp_tolerance:
            #                     print(f"System cooled to T_0 at t = {current_time:.2f}s")
            #                     break
                                
            # #                 if iteration >= max_iterations:
            # #                     print(f"Warning: Max iterations reached (t = {current_time:.2f}s)")
            # #                     break
                                
            #                 iteration += 1

            # Cool to ref
            self.T_n1.interpolate(self.T_0)
            thermal_strain_solve(self, self.T_n1, self.T_n)
            distortion_solve(self)

            self.T_n.assign(self.T_n1)
            self.tempFile.write(self.T_n)

            print("Max temp {:.2f}".format(np.max(self.T_n.vector().get_local())))

            ###### Mechanical Load Compliance ######
            u = fd.TrialFunction(self.displace)
            v = fd.TestFunction(self.displace)

            # Surface traction
            T = fd.conditional(
                fd.And(fd.gt(y, 0.095), fd.gt(x, 0.022)),
                fd.as_vector([-3e8, 0]),
                fd.as_vector([0, 0]))
            self.E.interpolate(self.E_0 + (self.E_1 - self.E_0) * (self.rho_hat ** self.penalisationExponent))

            lambda_ = (self.E * self.nu) / ((1 + self.nu) * (1 - 2 * self.nu))
            mu = (self.E) / (2 * (1 + self.nu))

            def epsilon(u): return 0.5 * (fd.grad(u) + fd.grad(u).T)
            def sigma(u): return lambda_ * fd.div(u) * Id + 2 * mu * epsilon(u)

            # Weak form
            a = fd.inner(sigma(u), epsilon(v)) * fd.dx
            L = fd.dot(T, v) * fd.ds(4)

            # Solve
            u = fd.Function(self.displace)
            fd.solve(a == L, u, bcs=[self.mech_bc])
            self.uFunction.assign(u)
            self.uFile.write(self.uFunction)

            # print("Type of u:", type(self.uFunction))
            # print("Shape of u:", self.uFunction.ufl_shape)
            # print("Type of u_t:", type(self.u_tFunction))
            # print("Shape of u_t:", self.u_tFunction.ufl_shape)

            self.stressintegral = 1

            # sigma_xx = self.stress_mech[0, 0]
            # von_mises_total_proj = fd.project(fd.sqrt(sigma_xx**2), self.densityspace)

            # # sigma_yy = self.stress_mech[1, 1]
            # # sigma_xy = self.stress_mech[0, 1]
            # # # DG0_scalar = fd.FunctionSpace(self.densityspace, "DG", 0)
            # # von_mises_stress = fd.sqrt(sigma_xx**2 - sigma_xx * sigma_yy + sigma_yy**2 + 3 * sigma_xy**2)
            
            # von_mises_total_proj = fd.project(fd.sqrt(sigma_xx**2), self.densityspace)
            # # print("max_stress", np.max(von_mises_total_proj.vector().get_local()))
            # self.von_mises_stress.assign(von_mises_total_proj)
            # self.von_mises_stressFile.write(self.von_mises_stress)

            # u_tx = self.u_tFunction[0]
            # u_ty = self.u_tFunction[1]
            # u_t = fd.sqrt(u_tx**2 + u_ty**2)
            # u_tproject = fd.project(u_t, self.densityspace)

            if self.scenario == 1:
                ##### Objective: Compliance #####
                self.j = fd.assemble(fd.inner(T, u) * self.ds(4))
                print(f"j = {self.j:.3g}")

                ##### Constraints #####
                volume_fraction = (1 / self.meshVolume) * fd.assemble(self.rho_hat * self.dx)
                self.c1 = self.Vlimit - volume_fraction
                self.c2 = 0

                ##### Objective sensitivities #####
                self.djdrho = (fda.compute_gradient(self.j, fda.Control(self.rho)).vector().get_local()) / self.gradientScale
                # print("djdrho ", self.djdrho)
                # print("djdrho normalised ", self.djdrho / np.max(np.abs(self.djdrho)))

                ##### Constraint sensitivities #####
                self.dc1drho = (fda.compute_gradient(self.c1, fda.Control(self.rho)).vector().get_local()) / self.gradientScale

                self.c = np.array([self.c1])
                self.dcdrho = self.dc1drho

                ##### Exploring thermal sensitivities #####
                # mech_stress_form = fd.inner(self.stress_mech, self.stress_mech) * self.dx
                # mech_stress_sensitivity_func = fd.Function(DG0_scalar)
                # gradient = fda.compute_gradient(fd.assemble(von_mises_total_proj * self.dx), fda.Control(self.rho))
                # grad_func = fd.Function(self.rho.function_space())
                # grad_func.vector()[:] = gradient.vector().get_local()

                # direction = fd.Constant((1.0, 0.0))

                # thermalstressintegral = fd.assemble(((von_mises_total_proj ** self.pnorm) * self.dx) ) ** (1/self.pnorm)
                # sen_thermalstressintegral = (fda.compute_gradient(thermalstressintegral, fda.Control(self.rho)).vector().get_local()) / self.gradientScale
                # max_abs_sensitivity = np.max(np.abs(sen_thermalstressintegral))
                # sen_thermalstressintegral = sen_thermalstressintegral / max_abs_sensitivity
                # print("thermal stress integral ", thermalstressintegral)
                # print("sensitivity ", sen_thermalstressintegral)

                coordinates = self.mesh.coordinates.dat.data_ro
                # plt.figure(figsize=(6, 6))
                # sc = plt.scatter(coordinates[:, 0], coordinates[:, 1], c=sen_thermalstressintegral, cmap='coolwarm', s=10)
                # plt.colorbar(sc, label='Normalised Sensitivity (dMech Stress Von Mises/drho)')
                # plt.show()

                # distortionintegral = fd.assemble(u_tproject * self.dx)
                # sen_distortionintegral = (fda.compute_gradient(distortionintegral, fda.Control(self.rho)).vector().get_local()) / self.gradientScale
                # print("sensitivity ", sen_distortionintegral)

                # plt.figure(figsize=(6, 6))
                # sc = plt.scatter(coordinates[:, 0], coordinates[:, 1], c=sen_distortionintegral, cmap='coolwarm', s=10)
                # plt.colorbar(sc, label='Normalised Sensitivity (dDistortion/drho)')
                # plt.show()

                # self.j_t = fd.assemble(fd.inner(T, self.u_tFunction) * self.ds(4))
                # self.j_t = fd.assemble(self.u_tFunction[0] * self.dx)
                # # print(f"j_t = {self.j_t:.3g}")
                # self.dj_tdrho = (fda.compute_gradient(self.j_t, fda.Control(self.rho)).vector().get_local()) / self.gradientScale
                # # self.dj_tdrho = self.dj_tdrho / np.max(np.abs(self.dj_tdrho))
                # print("dj_tdrho ", self.dj_tdrho)
                # plt.figure(figsize=(4, 6))
                # sc = plt.scatter(coordinates[:, 0], coordinates[:, 1], c=self.dj_tdrho, cmap='coolwarm', s=10)
                # plt.colorbar(sc, label='Sensitivity')
                # plt.show()


                        # self.j_tT = fd.assemble(self.stress_mech[0, 0] * self.dx)
                        # # print(f"j_tT = {self.j_tT:.3g}")
                        # self.dj_tTdrho = (fda.compute_gradient(self.j_tT, fda.Control(self.rho)).vector().get_local()) / self.gradientScale
                        # # self.dj_tTdrho = self.dj_tTdrho / np.max(np.abs(self.dj_tTdrho))
                        # print("dj_tTdrho ", self.dj_tTdrho)
                # plt.figure(figsize=(4, 6))
                # sc = plt.scatter(coordinates[:, 0], coordinates[:, 1], c=self.dj_tTdrho, cmap='coolwarm', s=10)
                # plt.colorbar(sc, label='Sensitivity')
                # plt.show()
                # self.j_tFile.write(self.dj_tTdrho)

                # # Compute directional sensitivity
                # directional_integrand = fd.dot(self.u_tFunction, direction) * self.dx
                # directional_sensitivity = fda.compute_gradient(fd.assemble(directional_integrand), fda.Control(self.rho)) / self.gradientScale

                # # Store and output
                # directional_sensitivity_func = fd.Function(DG0)
                # directional_sensitivity_func.vector()[:] = directional_sensitivity.vector().get_local()
                # self.mech_stress_sensitivityFile.write(directional_sensitivity_func)

                # Project to target space
                # mech_stress_sensitivity_func = fd.project(grad_func/self.gradientScale, DG0_scalar)
                # self.mech_stress_sensitivityFile.write(mech_stress_sensitivity_func)
                # c_thermal_form = fd.inner(self.u_tFunction, self.u_tFunction) * self.dx
                # c_thermal_sensitivity = (fda.compute_gradient(fd.assemble(c_thermal_form), fda.Control(self.rho)).vector().get_local()) / self.gradientScale
                # print(f"thermal: max={thermal_strain_sensitivity.max():.3e}, mean={thermal_strain_sensitivity.mean():.3e}\t" +
                #     f"mech: max={mech_stress_sensitivity.max():.3e}, mean={mech_stress_sensitivity.mean():.3e}\t" +
                #     f"disp: max={c_thermal_sensitivity.max():.3e}, mean={c_thermal_sensitivity.mean():.3e}")
                # self.mech_stress_sensitivityFun = fd.project(mech_stress_sensitivity, self.DG0)
                # nan, nan, nan, ..., nan, nan, nan
                
                # print(f"{thermal_strain_sensitivity:.3e}\t{mech_stress_sensitivity:.3e}\t{c_thermal_sensitivity:3e}\n")

            else:
                pass 
                
            with open(self.outputFolder2 + "combined_iteration_results.txt", "a") as log_file:
                total_time = time.time() - start_time
                log_file.write(f"{self.j:.3e}\t{volume_fraction:.3e}\t{self.c1:3e}\t{self.c2:.3e}\t{self.stressintegral:.3e}\t{total_time:.3e}\n")

        else:
            pass

        return self.j, self.djdrho, self.c, self.dcdrho
