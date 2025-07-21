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
        self.E_0, self.E_1 = 0.001, 1.0e11
        self.nu = 0.3

        # thermal properties
        self.T_0, self.T_1 = 400.0, 2000.0 # atmospheric temperature, heating temperature
        self.k_0, self.k_1 = 0.026, 17.0 # thermal conduction coeff of air, titanium
        self.alpha_0, self.alpha_1 = 1e-12, 8e-6 # linear thermal expansion coeff of air, titanium
        self.h = 100.0 # thermal convection coeff
        self.cp_0, self.cp_1 = 1e3, 500 # coeff specific heat capacity of air, titanium
        self.rho_material = 4500.0
        self.t_heat = 30.0 # time between deposit and next layer
        self.dt = 5.0 # timestep for t_heat
        self.dt2 = 20.0 # timestep for cooling
        
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
        self.nx, self.ny = 100, 200
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
        
        # self.mech_bc1 = fd.DirichletBC(self.displace, fd.Constant((0.0, 0.0)), 3) 
        # self.mech_bc2 = fd.DirichletBC(self.displace, fd.Constant((0.0, 0.0)), 2) 
        # self.mech_bc3 = fd.DirichletBC(self.displace, fd.Constant((0.0, 0.0)), 1) 
        # self.mech_bc4 = fd.DirichletBC(self.displace, fd.Constant((0.0, 0.0)), 4) 

        # Do collection of points
        self.mech_bc1 = FixedDirichletBC(self.displace, fd.Constant((0.0, 0.0)), np.array([0.025, 0.0]))
        self.mech_bc2 = FixedDirichletBC(self.displace, fd.Constant((0.0, 0.0)), np.array([0.026, 0.0]))
        self.mech_bc3 = FixedDirichletBC(self.displace, fd.Constant((0.0, 0.0)), np.array([0.024, 0.0]))
        self.temp_bc1 = fd.DirichletBC(self.templace, fd.Constant(self.T_0), 3)

        # output files
        self.rho_hatFile = VTKFile(self.outputFolder + "rho_hat.pvd")
        self.rho_hatFunction =fd.Function(self.densityspace, name="rho_hat")

        self.rho_dep =fd.Function(self.densityspace, name="rho_dep") # deposited rho
        self.rho_depFile = VTKFile(self.outputFolder + "rho_dep.pvd")

        self.alpha = fd.Function(self.densityspace, name="alpha")
        self.k_rho = fd.Function(self.densityspace, name="k_rho") # density-dependent thermal conductivity
        self.cp = fd.Function(self.densityspace, name="cp") # density-dependent thermal storage
        self.E = fd.Function(self.densityspace, name="E")
        self.alphaFile = VTKFile(self.outputFolder + "alpha.pvd")
        self.k_rhoFile = VTKFile(self.outputFolder + "k_rho.pvd")
        self.cpFile = VTKFile(self.outputFolder + "cp.pvd")
        self.EFile = VTKFile(self.outputFolder + "E.pvd")

        self.tempFile = VTKFile(self.outputFolder + "temp.pvd")
        self.T_n = fd.Function(self.templace, name="Temperature")
        self.T_n1 = fd.Function(self.templace)
        
        self.strain_th_increment = fd.Function(self.DG0, name="strain_th_increment")
        self.strain_th_incrementFile = VTKFile(self.outputFolder + "strain_th_increment.pvd")

        self.strain_therm = fd.Function(self.DG0, name="3strain_therm")
        self.strain_thermFile = VTKFile(self.outputFolder + "3strain_therm.pvd")

        self.total_strain = fd.Function(self.DG0, name="strain_total")
        self.total_strainFile = VTKFile(self.outputFolder + "1strain_total.pvd")

        self.strain_mech = fd.Function(self.DG0, name="strain_mech")
        self.strain_mechFile = VTKFile(self.outputFolder + "2strain_mech.pvd")

        self.stress_mech = fd.Function(self.DG0, name="stress_mech")
        self.stress_mechFile = VTKFile(self.outputFolder + "stress_mech.pvd")
    
        # mechanical distortion displacement
        # self.uFile = VTKFile(self.outputFolder + "u.pvd")
        # self.uFunction = fd.Function(self.displace, name="u")

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
            self.rho_hatFile.write(self.rho_hatFunction)
            
            ##### Temperature ######
            # Total strain = strain_mech + strain_therm
            def strain_total(u): return 0.5 * (fd.grad(u) + fd.grad(u).T)
            self.Id = fd.Identity(self.mesh.geometric_dimension())

            def print_strain_tensors(total_strain, mech_strain, therm_strain):
                def get_components(strain):
                    data = strain.dat.data[0]
                    return {
                        'xx': f"{data[0,0]:.3e}",
                        'xy': f"{data[0,1]:.3e}",
                        'yx': f"{data[1,0]:.3e}",
                        'yy': f"{data[1,1]:.3e}"
                    }
                
                t = get_components(total_strain)
                m = get_components(mech_strain)
                th = get_components(therm_strain)
                
                print(
                    f"Total [[{t['xx']}, {t['xy']}]    Mech [[{m['xx']}, {m['xy']}]    Therm [[{th['xx']}, {th['xy']}]\n"
                    f"       [{t['yx']}, {t['yy']}]          [{m['yx']}, {m['yy']}]            [{th['yx']}, {th['yy']}]]"
                )
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

            def thermal_strain_solve(self, T_n1, T_n, alpha):
                thermal_strain = alpha * (T_n1 - T_n) * self.rho_dep * self.Id
                self.strain_th_increment.interpolate(fd.project(thermal_strain, self.DG0))
                self.strain_th_incrementFile.write(self.strain_th_increment)
                self.strain_th_increment.interpolate(0.5 * self.strain_th_increment * (1 + fd.tanh(k * (self.rho_dep - 0.5))))
                self.strain_therm.interpolate((self.strain_th_increment + self.strain_therm) * self.rho_dep)
                self.strain_therm.interpolate(0.5 * self.strain_therm * (1 + fd.tanh(k * (self.rho_dep - 0.5))))
                self.strain_thermFile.write(self.strain_therm)

            def distortion_solve(self):
                rho = self.rho_dep

                u_t = fd.TrialFunction(self.displace)
                v_t = fd.TestFunction(self.displace)

                a_t = fd.inner((self.E * strain_total(u_t)), strain_total(v_t)) * rho * fd.dx
                L_t = fd.inner((self.E * self.strain_therm), strain_total(v_t)) * rho * fd.dx

                sol = fd.Function(self.displace)
                fd.solve(a_t == L_t, sol, bcs=[self.mech_bc1, self.mech_bc2, self.mech_bc3], solver_parameters=distortion_solver_parameters)
                sol.interpolate(0.5 * sol * (1 + fd.tanh(k * (rho - 0.5))))
                self.u_tFunction.interpolate(sol)
                self.u_tFile.write(self.u_tFunction)

                self.total_strain.interpolate(fd.project(strain_total(sol) * rho, self.DG0))
                self.total_strain.interpolate(0.5 * self.total_strain * (1 + fd.tanh(k * (rho - 0.5))))
                self.total_strainFile.write(self.total_strain) 

                self.strain_mech.interpolate(self.total_strain - self.strain_therm) 
                self.strain_mech.interpolate(0.5 * self.strain_mech * (1 + fd.tanh(k * (rho - 0.5))))
                self.strain_mechFile.write(self.strain_mech)
                # mu = self.E / (2 * (1 + self.nu))
                # lmbda = self.E * self.nu / (1 - self.nu**2) # Plane stress
                # self.stress_mech.interpolate(lmbda * fd.tr(self.strain_mech) * self.Id + 2 * mu * self.strain_mech)
                # self.stress_mech.interpolate(0.5 * self.stress_mech * (1 + fd.tanh(k * (rho - 0.5))))
                self.stress_mech.interpolate(self.E * self.strain_mech)
                self.stress_mechFile.write(self.stress_mech)

                print("Max temp {:.2f}".format(np.max(self.T_n.vector().get_local())))
          
            current_time = 0.0
            time_steps = []
            max_temps   = []
            max_strain_to  = []
            max_strain_me  = []
            max_strain_th  = []
            avg_strain_to  = []
            avg_strain_me  = []
            avg_strain_th  = []

            self.T_n.interpolate(fd.Constant(self.T_0))
            self.tempFile.write(self.T_n)
            self.T_n1.interpolate(fd.Constant(self.T_0))

            time_steps.append(0)
            max_temps.append(np.max(self.T_n.vector().get_local()))
            max_strain_to.append(0)
            max_strain_me.append(0)
            max_strain_th.append(0)
            avg_strain_to.append(0)
            avg_strain_me.append(0)
            avg_strain_th.append(0)

            n = 4
            layer_height = self.ly / n
            current_global_time = 0.0

            for i in range(n):
                print("i = ", i)
                layer_height_dep = layer_height * (i + 1)
                k = 10000

                self.rho_dep.interpolate(0.5 * (1 + fd.tanh(k * ( - y + layer_height_dep))) * self.rho_hat)
                self.rho_depFile.write(self.rho_dep)

                # self.alpha.interpolate(0.5 * (self.alpha_1 - self.alpha_0) * (1 + fd.tanh(k * (self.rho_dep - 0.5))) + self.alpha_0)
                # self.cp.interpolate(0.5 * (self.rho_material * self.cp_1 - 1.2 * self.cp_0) * (1 + fd.tanh(k * (self.rho_dep - 0.5))) + 1.2 * self.cp_0)
                # self.k_rho.interpolate(0.5 * (self.k_1 - self.k_0) * (1 + fd.tanh(k * (self.rho_dep - 0.5))) + self.k_0)
                # self.E.interpolate(0.5 * (self.E_1 - self.E_0) * (1 + fd.tanh(k * (self.rho_dep - 0.5))) + self.E_0)
                
                self.alpha.interpolate(self.alpha_0 + (self.alpha_1 - self.alpha_0) * (self.rho_dep ** self.penalisationExponent))
                self.cp.interpolate(1.2 * self.cp_0 + (self.rho_material * self.cp_1 - 1.2 * self.cp_0) * (self.rho_dep ** self.penalisationExponent))
                self.k_rho.interpolate(self.k_0 + (self.k_1 - self.k_0) * (self.rho_dep ** self.penalisationExponent))
                self.E.interpolate(self.E_0 + (self.E_1 - self.E_0) * (self.rho_dep ** self.penalisationExponent))  

                self.T_n1.interpolate(self.T_n * (1 + 0.5 * (fd.tanh(k * ( - y + layer_height_dep -layer_height)) + fd.tanh(k * (y - layer_height_dep))) * 0.5 * (1 + fd.tanh(k * (self.rho_hat - 0.5)))) 
                                    + self.T_1 * (-0.5*(fd.tanh(k*(-y+layer_height_dep-layer_height))+fd.tanh(k*(y-layer_height_dep))))*(0.5*(1+fd.tanh(k*(self.rho_hat - 0.5))))) 
                
                self.alphaFile.write(self.alpha)
                self.cpFile.write(self.cp)
                self.k_rhoFile.write(self.k_rho)
                self.EFile.write(self.E)
                self.T_n.interpolate(self.T_n1)
                self.tempFile.write(self.T_n)

                current_time = 0.0

                current_global_time += self.dt
                time_steps.append(current_global_time)
                max_temps.append(np.max(self.T_n.vector().get_local()))

                max_strain_to.append(0)
                max_strain_me.append(0)
                max_strain_th.append(0)
                avg_strain_to.append(0)
                avg_strain_me.append(0)
                avg_strain_th.append(0)
            
                while current_time < self.t_heat:
                    w = fd.TestFunction(self.templace)

                    #### Temperature Solve ####
                    F = (self.cp * (self.T_n1 - self.T_n) * w * self.dx + self.dt * self.k_rho * fd.dot(fd.grad(self.T_n1), fd.grad(w)) * self.dx)
                    F += self.dt * self.h * (self.T_n1 - self.T_0) * w * self.ds(1)
                    F += self.dt * self.h * (self.T_n1 - self.T_0) * w * self.ds(2)
                    F += self.dt * self.h * (self.T_n1 - self.T_0) * w * self.ds(4)
                    fd.solve(F == 0, self.T_n1, bcs=[self.temp_bc1])
                    
                    # Here, new temp is T_n1, previous temp is T_n
                    thermal_strain_solve(self, self.T_n1, self.T_n, self.alpha)
                    distortion_solve(self)

                    self.T_n.assign(self.T_n1)
                    self.tempFile.write(self.T_n)

                    current_time += self.dt

                    current_global_time += self.dt
                    time_steps.append(current_global_time)
                    max_temps.append(np.max(self.T_n1.vector().get_local()))
                    max_strain_to.append(np.max(np.abs(self.total_strain.dat.data)))
                    max_strain_me.append(np.max(np.abs(self.strain_mech.dat.data)))
                    max_strain_th.append(np.max(np.abs(self.strain_therm.dat.data)))
                    avg_strain_to.append(np.mean(np.abs(self.total_strain.dat.data)))
                    avg_strain_me.append(np.mean(np.abs(self.strain_mech.dat.data)))
                    avg_strain_th.append(np.mean(np.abs(self.strain_therm.dat.data)))

                    if i == n - 1: # Forced cooled to ambient
                        temp_tolerance = 0.5  # Temperature difference considered "cooled"
                        max_iterations = 500   # Safety limit to prevent infinite loops
                        iteration = 0
                        while True:
                            w = fd.TestFunction(self.templace)

                            #### Temperature Solve ####
                            F = (self.cp * (self.T_n1 - self.T_n) * w * self.dx + self.dt * self.k_rho * fd.dot(fd.grad(self.T_n1), fd.grad(w)) * self.dx)
                            F += self.dt * self.h * (self.T_n1 - self.T_0) * w * self.ds(1)
                            F += self.dt * self.h * (self.T_n1 - self.T_0) * w * self.ds(2)
                            F += self.dt * self.h * (self.T_n1 - self.T_0) * w * self.ds(4)
                            fd.solve(F == 0, self.T_n1, bcs=[self.temp_bc1])
                            
                            # Here, new temp is T_n1, previous temp is T_n
                            thermal_strain_solve(self, self.T_n1, self.T_n, self.alpha)
                            distortion_solve(self)

                            self.T_n.assign(self.T_n1)
                            self.tempFile.write(self.T_n)

                            current_time += self.dt

                            current_global_time += self.dt
                            time_steps.append(current_global_time)
                            max_temps.append(np.max(self.T_n1.vector().get_local()))
                            max_strain_to.append(np.max(np.abs(self.total_strain.dat.data)))
                            max_strain_me.append(np.max(np.abs(self.strain_mech.dat.data)))
                            max_strain_th.append(np.max(np.abs(self.strain_therm.dat.data)))
                            avg_strain_to.append(np.mean(np.abs(self.total_strain.dat.data)))
                            avg_strain_me.append(np.mean(np.abs(self.strain_mech.dat.data)))
                            avg_strain_th.append(np.mean(np.abs(self.strain_therm.dat.data)))

                            temp_diff = np.max(self.T_n.vector().get_local()) - self.T_0
                            
                            # Temp convergence
                            if temp_diff < temp_tolerance:
                                print(f"System cooled to T_0 at t = {current_time:.2f}s")
                                break
                                
                            if iteration >= max_iterations:
                                print(f"Warning: Max iterations reached (t = {current_time:.2f}s)")
                                break
                                
                            iteration += 1

            plt.figure(figsize=(15, 5))

            plt.subplot(1, 3, 1)
            plt.plot(time_steps, max_temps, 'r-')
            plt.xlabel('Time [s]')
            plt.ylabel('Max temperature [K]')
            plt.grid(True)

            plt.subplot(1, 3, 2)
            plt.plot(time_steps, max_strain_to, 'b-', label='Max Total Strain')
            plt.plot(time_steps, max_strain_me, 'g--', label='Max Mechanical Strain')
            plt.plot(time_steps, max_strain_th, 'm:', label='Max Thermal Strain')
            plt.xlabel('Time [s]')
            plt.ylabel('Max Strain')
            plt.grid(True)
            plt.legend()

            plt.subplot(1, 3, 3)
            plt.plot(time_steps, avg_strain_to, 'b-', label='Avg Total Strain')
            plt.plot(time_steps, avg_strain_me, 'g--', label='Avg Mechanical Strain')
            plt.plot(time_steps, avg_strain_th, 'm:', label='Avg Thermal Strain')
            plt.xlabel('Time [s]')
            plt.ylabel('Average Strain')
            plt.grid(True)
            plt.legend()

            plt.tight_layout()
            plt.show()

            # plt.figure(figsize=(12, 5))
            # plt.subplot(1, 2, 1)
            # plt.semilogx(time_steps, max_temps, 'r-')
            # plt.xlabel('Time [s] (log scale)')
            # plt.ylabel('Max temperature [K]')
            # plt.grid(True, which="both", ls="-")

            # plt.subplot(1, 2, 2)
            # plt.semilogx(time_steps, max_strain_to, 'b-', label='Total Strain')
            # plt.semilogx(time_steps, max_strain_me, 'g--', label='Mechanical Strain') 
            # plt.semilogx(time_steps, max_strain_th, 'm:', label='Thermal Strain')
            # plt.xlabel('Time [s] (log scale)')
            # plt.ylabel('Max Strain')
            # plt.grid(True, which="both", ls="-")
            # plt.legend()

            # plt.tight_layout()
            # plt.show()

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
