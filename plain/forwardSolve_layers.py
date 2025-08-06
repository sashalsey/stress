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
        self.nu = 0.0

        # thermal properties
        self.T_0, self.T_1 = 0.0, 100.0 # atmospheric temperature, heating temperature
        self.k_0, self.k_1 = 0.026, 17.0 # thermal conduction coeff of air, titanium
        self.alpha_0, self.alpha_1 = 1e-12, 8e-6 # linear thermal expansion coeff of air, titanium
        self.h = 100.0 # thermal convection coeff
        self.cp_0, self.cp_1 = 1e3, 500 # coeff specific heat capacity of air, titanium
        self.rho_material = 4500.0
        self.t_heat = 5.0 # time between deposit and next layer
        self.dt = 1.0 # timestep for t_heat
        
        # pseudo-density functional
        self.beta = beta  # heaviside projection paramesteepnesster
        self.eta0 = 0.5  # midpoint of projection filter
        
        self.Vlimit = 1.0 # volume fraction constraint
        self.pnorm = pnorm # p-norm stress penalty
        self.initial_stressintegral = float(initial_stressintegral)

    def GenerateMesh(self,):
        self.mesh = fd.RectangleMesh(self.nx, self.ny, self.lx, self.ly, quadrilateral=True)
        self.gradientScale = (self.nx * self.ny) / (self.lx * self.ly)
        self.dx = fd.Measure('dx', domain=self.mesh)
        self.ds = fd.ds(domain=self.mesh)
        self.n = fd.FacetNormal(self.mesh)

    def Setup(self):
        self.nx, self.ny = 200, 100
        self.lx, self.ly =  0.1, 0.05
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
        self.mech_bc1 = FixedDirichletBC(self.displace, fd.Constant((0.0, 0.0)), np.array([0.05, 0.0]))
        self.mech_bc2 = FixedDirichletBC(self.displace, fd.Constant((0.0, 0.0)), np.array([0.049, 0.0]))
        self.mech_bc3 = FixedDirichletBC(self.displace, fd.Constant((0.0, 0.0)), np.array([0.051, 0.0]))
        self.temp_bc1 = fd.DirichletBC(self.templace, fd.Constant(0.0), 3)

        # output files
        self.rho_hatFunction =fd.Function(self.densityspace, name="rho_hat")
        # self.rho_hatFile = VTKFile(self.outputFolder + "rho_hat.pvd")
        self.rho_dep =fd.Function(self.densityspace, name="rho_dep") # deposited rho 
        self.E = fd.Function(self.densityspace, name="E")
        self.k_rho = fd.Function(self.densityspace, name="k_rho") # density-dependent thermal conductivity
        self.cp = fd.Function(self.densityspace, name="cp") # density-dependent thermal storage
        self.alpha = fd.Function(self.densityspace, name="alpha")
        self.rho_depFile = VTKFile(self.outputFolder + "rho_dep.pvd")
        self.k_rhoFile = VTKFile(self.outputFolder + "k_rho.pvd")
        self.cpFile = VTKFile(self.outputFolder + "cp.pvd")
        self.alphaFile = VTKFile(self.outputFolder + "alpha.pvd")
        self.EFile = VTKFile(self.outputFolder + "E.pvd")

        self.tempFile = VTKFile(self.outputFolder + "temp.pvd")
        self.T_n = fd.Function(self.templace, name="Temperature")
        self.T_n1 = fd.Function(self.templace)
        
        self.strain_therm_increment = fd.Function(self.DG0, name="strain_therm_increment")
        self.strain_therm_incrementFile = VTKFile(self.outputFolder + "strain_therm_increment.pvd")

        self.strain_therm = fd.Function(self.DG0, name="strain_therm")
        self.strain_thermFile = VTKFile(self.outputFolder + "strain_therm.pvd")

        self.strain_total_increment = fd.Function(self.DG0, name="strain_total_increment")
        self.strain_total_incrementFile = VTKFile(self.outputFolder + "strain_total_increment.pvd")

        self.strain_total = fd.Function(self.DG0, name="strain_total")
        self.strain_totalFile = VTKFile(self.outputFolder + "strain_total.pvd")

        self.strain_mech_increment = fd.Function(self.DG0, name="strain_mech_increment")
        self.strain_mech_incrementFile = VTKFile(self.outputFolder + "strain_mech_increment.pvd")

        self.strain_mech = fd.Function(self.DG0, name="strain_mech")
        self.strain_mechFile = VTKFile(self.outputFolder + "strain_mech.pvd")

        self.stress_mech = fd.Function(self.DG0, name="stress_mech")
        self.stress_mechFile = VTKFile(self.outputFolder + "stress_mech.pvd")
    
        # mechanical distortion displacement
        self.uFile = VTKFile(self.outputFolder + "u.pvd")
        self.uFunction = fd.Function(self.displace, name="u")

        # thermal distortion displacement
        self.u_tFile = VTKFile(self.outputFolder + "u_t.pvd")
        self.u_tFunction = fd.Function(self.displace, name="u_t")

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
            # self.rho_hatFile.write(self.rho_hatFunction)
            
            ##### Temperature ######
            # Total strain = strain_mech + strain_therm
            def total_strain_def(u): return 0.5 * (fd.grad(u) + fd.grad(u).T)

            def thermal_strain_solve(self, T_n1, T_n):
                thermal_strain = self.alpha * (T_n1 - T_n) * self.rho_dep * Id
                self.strain_therm_increment.interpolate(fd.project(thermal_strain, self.DG0))

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
                a_t = fd.inner((self.E * total_strain_def(u_t)), total_strain_def(v_t))* self.rho_dep * fd.dx
                L_t = fd.inner((self.E * self.strain_therm_increment), total_strain_def(v_t))* self.rho_dep * fd.dx
                sol = fd.Function(self.displace)
                fd.solve(a_t == L_t, sol, bcs=[self.mech_bc1, self.mech_bc2, self.mech_bc3], solver_parameters=distortion_solver_parameters)
                
                self.u_tFunction.interpolate(self.u_tFunction + sol * self.rho_dep)
                self.strain_total_increment.interpolate(fd.project(total_strain_def(sol) * self.rho_dep, self.DG0))
                self.strain_total.interpolate(self.strain_total + self.strain_total_increment)
                self.strain_mech_increment.interpolate(self.strain_total_increment - self.strain_therm_increment)
                self.strain_mech.interpolate(self.strain_mech + self.strain_mech_increment)

                self.u_tFile.write(self.u_tFunction)
                self.strain_total_incrementFile.write(self.strain_total_increment)
                self.strain_totalFile.write(self.strain_total)
                self.strain_mech_incrementFile.write(self.strain_mech_increment)
                self.strain_mechFile.write(self.strain_mech)
                self.stress_mech.interpolate(self.E * self.strain_mech)
                self.stress_mechFile.write(self.stress_mech)
            
            def flatten_stuff(data_):
                # data = data_.dat.data
                flattened_data = data_.flatten()
                index = np.argmax(np.abs(flattened_data))
                max_magnitude_value = flattened_data[index]
                # mean_magnitude_value = np.mean(flattened_data)
                return max_magnitude_value

            def store_stuff(self):
                max_temps.append(np.max(self.T_n.vector().get_local()))
                max_strain_to.append(flatten_stuff(self.strain_total.dat.data))
                max_strain_me.append(flatten_stuff(self.strain_mech.dat.data))
                max_strain_th.append(flatten_stuff(self.strain_therm.dat.data))
                avg_strain_to.append(np.mean(self.strain_total.dat.data))
                avg_strain_me.append(np.mean(self.strain_mech.dat.data))
                avg_strain_th.append(np.mean(self.strain_therm.dat.data))

                max_strain_toi.append(flatten_stuff(self.strain_total_increment.dat.data))
                max_strain_mei.append(flatten_stuff(self.strain_mech_increment.dat.data))
                max_strain_thi.append(flatten_stuff(self.strain_therm_increment.dat.data))
                avg_strain_toi.append(np.mean(self.strain_total_increment.dat.data))
                avg_strain_mei.append(np.mean(self.strain_mech_increment.dat.data))
                avg_strain_thi.append(np.mean(self.strain_therm_increment.dat.data))

                u_x_bottom_right.append(self.u_tFunction((self.lx, 1 * self.cellsize))[0])
                u_y_bottom_right.append(self.u_tFunction((self.lx, 1 * self.cellsize))[1])
                u_x_mid_right.append(self.u_tFunction((self.lx, self.ly/2 - 1 * self.cellsize))[0])
                u_y_mid_right.append(self.u_tFunction((self.lx, self.ly/2 - 1 * self.cellsize))[1])
                u_x_top_right.append(self.u_tFunction((self.lx, self.ly - 1 * self.cellsize))[0])
                u_y_top_right.append(self.u_tFunction((self.lx, self.ly - 1 * self.cellsize))[1])
                # print("Bottom ", self.u_tFunction((self.lx, 1 * self.cellsize)))
                # print("Bottom mid ", self.u_tFunction((self.lx, self.ly/4)))
                # print("Mid ", self.u_tFunction((self.lx, self.ly/2 - 1 * self.cellsize)))
                # print("Top mid ", self.u_tFunction((self.lx, 3*self.ly/4)))
                # print("Top ", self.u_tFunction((self.lx, self.ly - 1 * self.cellsize)))
  
            Id = fd.Identity(self.mesh.geometric_dimension())
            current_time = 0.0
            time_steps, max_temps = [], []
            max_strain_to, max_strain_me, max_strain_th = [], [], []
            avg_strain_to, avg_strain_me, avg_strain_th = [], [], []
            max_strain_toi, max_strain_mei, max_strain_thi = [], [], []
            avg_strain_toi, avg_strain_mei, avg_strain_thi = [], [], []
            u_x_bottom_right, u_y_bottom_right, u_x_top_right, u_y_top_right, u_x_mid_right, u_y_mid_right = [], [], [], [], [], []

            self.T_n.interpolate(fd.Constant(self.T_0))
            self.tempFile.write(self.T_n)
            self.T_n1.interpolate(fd.Constant(self.T_0))

            n = 2
            layer_height = self.ly / n
            current_global_time = 0.0

            def layer_split(self):
                self.strain_mech_increment.dat.data
                layer_1 = fd.Function(self.DG0)
                layer_2 = fd.Function(self.DG0)
                # layer_1.interpolate(self.strain_mech_increment*(-0.5*(1 + fd.tanh(1000*(y - layer_height)))+1))
                # layer_2.interpolate(self.strain_mech_increment*(0.5*(1 + fd.tanh(1000*(y - layer_height)))))
                layer_1.interpolate(self.strain_mech_increment*(0.5*(1 + fd.tanh(1000*(y - 3*self.cellsize))))*(-0.5*(1 + fd.tanh(1000*(y - layer_height + 3*self.cellsize)))+1))
                layer_2.interpolate(self.strain_mech_increment*(0.5*(1 + fd.tanh(1000*(y - layer_height - 3*self.cellsize))))*(-0.5*(1 + fd.tanh(1000*(y - n*layer_height + 3*self.cellsize)))+1))
                print("layer 1 mean mech strain increment ", np.mean(layer_1.dat.data))
                print("layer 2 mean mech strain increment ", np.mean(layer_2.dat.data))
              
            # First layer deposited at 100
            layer_height_dep = layer_height * (1)

            self.rho_dep.interpolate(fd.conditional(fd.lt(y, layer_height_dep), self.rho_hat, 0))
            self.k_rho.interpolate(fd.conditional(fd.lt(y, layer_height_dep), self.k_1, self.k_0))
            self.alpha.interpolate(fd.conditional(fd.lt(y, layer_height_dep), self.alpha_1, self.alpha_0))
            self.cp.interpolate(fd.conditional(fd.lt(y, layer_height_dep), self.rho_material * self.cp_1, 1.2 * self.cp_0))
            self.E.interpolate(fd.conditional(fd.lt(y, layer_height_dep), self.E_1, self.E_0))
            self.rho_depFile.write(self.rho_dep)
            self.k_rhoFile.write(self.k_rho)
            self.alphaFile.write(self.alpha)
            self.cpFile.write(self.cp)
            self.EFile.write(self.E)

            # First layer deposited at 100
            self.T_n.interpolate(fd.conditional(
                    fd.And(fd.gt(y, (layer_height_dep - layer_height)), fd.le(y, layer_height_dep)),
                    100.0,
                    0.0 ))
            self.tempFile.write(self.T_n)

            time_steps.append(current_time)
            max_temps.append(np.max(self.T_n.vector().get_local()))
            for lst in [max_strain_to, max_strain_me, max_strain_th, avg_strain_to, avg_strain_me, avg_strain_th, u_x_bottom_right, u_y_bottom_right, u_x_top_right, u_y_top_right, 
                        max_strain_toi, max_strain_mei, max_strain_thi, avg_strain_toi, avg_strain_mei, avg_strain_thi, u_x_mid_right, u_y_mid_right]:
                lst.append(0)

            # First layer cools to 80
            self.T_n1.interpolate(fd.conditional(
                    fd.And(fd.gt(y, (layer_height_dep - layer_height)), fd.le(y, layer_height_dep)),
                    80.0,
                    0.0 ))

            ##### Distortion Solve 1 #####
            thermal_strain_solve(self, self.T_n1, self.T_n)
            distortion_solve(self)
            layer_split(self)

            self.T_n.interpolate(self.T_n1)
            self.tempFile.write(self.T_n)

            current_time += self.dt
            current_global_time += self.dt
            time_steps.append(current_global_time)
            store_stuff(self)
            
            # Second layer deposited at 100
            layer_height_dep = layer_height * (2)

            self.rho_dep.interpolate(fd.conditional(fd.le(y, layer_height_dep), self.rho_hat, 0))
            self.k_rho.interpolate(fd.conditional(fd.le(y, layer_height_dep), self.k_1, self.k_0))
            self.alpha.interpolate(fd.conditional(fd.le(y, layer_height_dep), self.alpha_1, self.alpha_0))
            self.cp.interpolate(fd.conditional(fd.le(y, layer_height_dep), self.rho_material * self.cp_1, 1.2 * self.cp_0))
            self.E.interpolate(fd.conditional(fd.le(y, layer_height_dep), self.E_1, self.E_0))
            self.rho_depFile.write(self.rho_dep)
            self.k_rhoFile.write(self.k_rho)
            self.alphaFile.write(self.alpha)
            self.cpFile.write(self.cp)
            self.EFile.write(self.E)

            bottom_layer_min = layer_height_dep - 2 * layer_height  # Bottom layer starts here
            bottom_layer_max = layer_height_dep - layer_height      # Bottom layer ends here
            top_layer_min = layer_height_dep - layer_height         # Top layer starts here
            top_layer_max = layer_height_dep                        # Top layer ends here

            # Second Layer deposited
            self.T_n.interpolate(
                fd.conditional(
                    fd.And(fd.gt(y, top_layer_min), fd.le(y, top_layer_max)),
                    100.0,  # Top layer
                    fd.conditional(
                        fd.And(fd.gt(y, bottom_layer_min), fd.le(y, bottom_layer_max)),
                        80.0,  # Bottom layer
                        0.0)))    # Everywhere else
            self.tempFile.write(self.T_n)

            current_time += self.dt
            current_global_time += self.dt
            time_steps.append(current_global_time)
            store_stuff(self)

            ##### LOOP #####
            Temps_1 = [100, 80, 60, 40, 20, 0, 0]
            Temps_2 = [80, 67, 51, 37, 20, 0, 0]
            for i in range(len(Temps_1) - 1):
                self.T_n1.interpolate(
                    fd.conditional(
                        fd.And(fd.gt(y, top_layer_min), fd.le(y, top_layer_max)),
                        Temps_1[i],  # Top layer
                        fd.conditional(
                            fd.And(fd.gt(y, bottom_layer_min), fd.le(y, bottom_layer_max)),
                            Temps_1[i+1],  # Bottom layer
                            0.0)))    # Everywhere else
                
                thermal_strain_solve(self, self.T_n1, self.T_n)
                distortion_solve(self)
                layer_split(self)

                self.T_n.interpolate(self.T_n1)
                self.tempFile.write(self.T_n)

                current_time += self.dt
                current_global_time += self.dt
                time_steps.append(current_global_time)
                store_stuff(self)

            # Cool to ref
            self.rho_dep.interpolate(self.rho_hat)
            self.alpha.interpolate(self.alpha_1)
            self.E.interpolate(self.E_1)
            self.rho_depFile.write(self.rho_dep)
            self.alphaFile.write(self.alpha)
            self.EFile.write(self.E)

            self.T_n1.interpolate(self.T_0)
            thermal_strain_solve(self, self.T_n1, self.T_n)
            distortion_solve(self)
            layer_split(self)

            self.T_n.assign(self.T_n1)
            self.tempFile.write(self.T_n)

            print("Max temp {:.2f}".format(np.max(self.T_n.vector().get_local())))

            current_time += self.dt
            current_global_time += self.dt
            time_steps.append(current_global_time)
            store_stuff(self)
            
            
            
            plt.figure(figsize=(15, 5))

            plt.subplot(1, 3, 1)
            plt.plot(time_steps, max_temps, 'r-')
            plt.xlabel('Time [s]')
            plt.ylabel('Max temperature [K]')
            plt.grid(True)

            plt.subplot(1, 3, 2)
            plt.plot(time_steps, avg_strain_toi, 'b-', label='Avg Total Strain Increment')
            plt.plot(time_steps, avg_strain_mei, 'g--', label='Avg Mechanical Increment')
            plt.plot(time_steps, avg_strain_thi, 'r:', label='Avg Thermal Strain Increment')
            plt.xlabel('Time [s]')
            plt.ylabel('Average Strain')
            plt.grid(True)
            plt.legend()

            plt.subplot(1, 3, 3)
            plt.plot(time_steps, avg_strain_to, 'b-', label='Avg Total Strain')
            plt.plot(time_steps, avg_strain_me, 'g--', label='Avg Mechanical Strain')
            plt.plot(time_steps, avg_strain_th, 'r:', label='Avg Thermal Strain')
            plt.xlabel('Time [s]')
            plt.ylabel('Average Strain')
            plt.grid(True)
            plt.legend()

            plt.tight_layout()
            plt.show()

            plt.figure()
            plt.plot(time_steps, u_x_bottom_right, 'b-', label='u_x_bottom_right')
            plt.plot(time_steps, u_y_bottom_right, 'g--', label='u_y_bottom_right')
            plt.plot(time_steps, u_x_top_right, 'm-', label='u_x_top_right')
            plt.plot(time_steps, u_y_top_right, 'r:', label='u_y_top_right')
            plt.xlabel('Time [s]')
            plt.ylabel('Displacement')
            plt.grid(True)
            plt.legend()

            plt.show()

            plt.figure()
            plt.plot(time_steps, avg_strain_me, 'g-', label='Avg Mechanical Strain')
            plt.plot(time_steps, avg_strain_mei, 'g:', label='Avg Mechanical Increment')
            plt.xlabel('Time [s]')
            plt.ylabel('Strain')
            plt.grid(True)
            plt.legend()

            plt.show()

            sys.exit()

            ###### Mechanical Load Compliance ######
            u = fd.TrialFunction(self.displace)
            v = fd.TestFunction(self.displace)

            # Surface traction
            T = fd.conditional(
                fd.And(fd.gt(y, 0.095), fd.gt(x, 0.04)),
                fd.as_vector([-3e8, 0]),
                fd.as_vector([0, 0]))

            # Weak form
            a = fd.inner(sigma(u), epsilon(v)) * fd.dx
            L = fd.dot(T, v) * fd.ds(4)

            # Solve
            u = fd.Function(self.displace)
            fd.solve(a == L, u, bcs=[self.mech_bc1])
            self.uFunction.assign(u)
            self.uFile.write(self.uFunction)

            self.stressintegral = 1

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

                ##### Constraint sensitivities #####
                self.dc1drho = (fda.compute_gradient(self.c1, fda.Control(self.rho)).vector().get_local()) / self.gradientScale

                self.c = np.array([self.c1])
                self.dcdrho = self.dc1drho

            else:
                pass 
                
            with open(self.outputFolder2 + "combined_iteration_results.txt", "a") as log_file:
                total_time = time.time() - start_time
                log_file.write(f"{self.j:.3e}\t{volume_fraction:.3e}\t{self.c1:3e}\t{self.c2:.3e}\t{self.stressintegral:.3e}\t{total_time:.3e}\n")

        else:
            pass

        return self.j, self.djdrho, self.c, self.dcdrho
