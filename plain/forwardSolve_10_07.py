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
        self.T_0, self.T_1 = 0.0, 100.0 # atmospheric temperature, heating temperature
        self.k_0, self.k_1 = 0.026, 17.0 # thermal conduction coeff of air, titanium
        self.alpha_0, self.alpha_1 = 2e-3, 8e-6 # linear thermal expansion coeff of air, titanium
        self.h = 100.0 # thermal convection coeff
        self.cp_0, self.cp_1 = 1e3, 500 # coeff specific heat capacity of air, titanium
        self.rho_material = 4500.0
        self.t_heat = 50.0 # time between deposit and next layer
        self.dt = 20.0 # timestep for t_heat
        
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
        self.nx, self.ny = 1, 1
        self.lx, self.ly =  0.1, 0.1
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
        self.mech_bc1 = FixedDirichletBC(self.displace.sub(1), fd.Constant((0.0)), np.array([0.0, 0.0]))
        # self.mech_bc1 = fd.DirichletBC(self.displace.sub(1), fd.Constant((0.0)), 3)
        # self.mech_bc2 = fd.DirichletBC(self.displace.sub(1), fd.Constant((0.0)), 4)
        # self.mech_bc3 = fd.DirichletBC(self.displace.sub(1), fd.Constant((0.0)), 1)
        # self.mech_bc4 = fd.DirichletBC(self.displace.sub(1), fd.Constant((0.0)), 2)
        self.temp_bc1 = fd.DirichletBC(self.templace, fd.Constant(0.0), 3)
        # self.temp_bc2 = fd.DirichletBC(self.templace, fd.Constant(0.0), 4)

        # output files
        self.rho_hatFile = VTKFile(self.outputFolder + "rho_hat.pvd")
        self.rho_hatFunction =fd.Function(self.densityspace, name="rho_hat")

        self.k_rho = fd.Function(self.densityspace, name="k_rho") # density-dependent thermal conductivity
        self.cp = fd.Function(self.densityspace, name="cp") # density-dependent thermal storage

        self.tempFile = VTKFile(self.outputFolder + "temp.pvd")
        self.T_n = fd.Function(self.templace, name="Temperature")
        self.T_n1 = fd.Function(self.templace)

        # self.thermal_strain = fd.Function(self.DG0)
        self.strain_th_increment = fd.Function(self.DG0)

        self.strain_therm = fd.Function(self.DG0, name="strain_therm")
        self.strain_thermFile = VTKFile(self.outputFolder + "strain_therm.pvd")

        self.total_strain = fd.Function(self.DG0, name="strain_total")
        self.total_strainFile = VTKFile(self.outputFolder + "strain_total.pvd")

        self.strain_mech = fd.Function(self.DG0, name="strain_mech")
        self.strain_mechFile = VTKFile(self.outputFolder + "strain_mech.pvd")

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
            def distortion_solve(T_n1, T_n):
                thermal_strain = self.alpha_1 * (T_n1 - T_n) * self.rho_hat * Id
                self.strain_th_increment.interpolate(fd.project(thermal_strain, self.DG0))
                self.strain_therm.interpolate(self.strain_th_increment + self.strain_therm)
                self.strain_thermFile.write(self.strain_therm)

                u_t = fd.TrialFunction(self.displace)
                v_t = fd.TestFunction(self.displace)

                self.E = self.E_0 + (self.E_1 - self.E_0) * (self.rho_hat ** self.penalisationExponent)

                a_t = fd.inner((self.E * strain_total(u_t)), strain_total(v_t)) * fd.dx
                L_t = fd.inner((self.E * self.strain_therm), strain_total(v_t)) * fd.dx

                sol = fd.Function(self.displace)
                fd.solve(a_t == L_t, sol, bcs=[self.mech_bc1], solver_parameters=distortion_solver_parameters)
                # , self.mech_bc2
                self.u_tFunction.interpolate(sol)
                self.u_tFile.write(self.u_tFunction)

                self.total_strain.interpolate(fd.project(strain_total(sol), self.DG0))
                self.total_strainFile.write(self.total_strain) 

                self.strain_mech.interpolate(self.total_strain - self.strain_therm) 
                self.strain_mechFile.write(self.strain_mech)
                mu = self.E / (2 * (1 + self.nu))
                lmbda = self.E * self.nu / (1 - self.nu**2) # Plane stress
                self.stress_mech.interpolate(lmbda * fd.tr(self.strain_mech) * Id + 2 * mu * self.strain_mech)
                self.stress_mechFile.write(self.stress_mech)

                print("Max temp {:.3e}".format(np.max(self.T_n.vector().get_local())))
                print("u_t max dat {:.4e}".format(np.max(np.abs(self.u_tFunction.dat.data))))

                print_strain_tensors(self.total_strain, self.strain_mech, self.strain_therm)


            Id = fd.Identity(self.mesh.geometric_dimension())
            self.k_rho.interpolate(fd.Constant(self.k_1))
            self.cp.interpolate(fd.Constant(self.rho_material * self.cp_1))
            self.T_n1.interpolate(fd.Constant(self.T_1))

            current_time = 0.0
            time_steps = []
            max_temps   = []
            max_strain_to  = []
            max_strain_me  = []
            max_strain_th  = []

            self.T_n.interpolate(self.T_n1)
            self.tempFile.write(self.T_n)

            time_steps.append(current_time)
            max_temps.append(np.max(self.T_n.vector().get_local()))
            max_strain_to.append(0)
            max_strain_me.append(0)
            max_strain_th.append(0)

            temp_tolerance = 0.01  # Temperature difference considered "cooled"
            max_iterations = 500   # Safety limit to prevent infinite loops
            iteration = 0

            while True:
                # Temperature Solve
                w = fd.TestFunction(self.templace)
                F = (self.cp * (self.T_n1 - self.T_n) * w * self.dx + self.dt * self.k_rho * fd.dot(fd.grad(self.T_n1), fd.grad(w)) * self.dx)
                fd.solve(F == 0, self.T_n1, bcs=[self.temp_bc1]) # , self.temp_bc2

                # Distortion Solve
                distortion_solve(self.T_n1, self.T_n)

                self.T_n.assign(self.T_n1)
                self.tempFile.write(self.T_n)

                # Calculate maximum temperature difference
                max_temp = np.max(self.T_n.vector().get_local())
                temp_diff = max_temp - self.T_0
                
                time_steps.append(current_time)
                max_temps.append(max_temp)
                max_strain_to.append(np.max(np.abs(self.total_strain.dat.data)))
                max_strain_me.append(np.max(np.abs(self.strain_mech.dat.data)))
                max_strain_th.append(np.max(np.abs(self.strain_therm.dat.data)))
                
                # Check convergence
                if temp_diff < temp_tolerance:
                    print(f"System cooled to T_0 at t = {current_time:.2f}s")
                    break
                    
                if iteration >= max_iterations:
                    print(f"Warning: Max iterations reached (t = {current_time:.2f}s)")
                    break
                    
                # Update counters
                current_time += self.dt
                iteration += 1

            # Final state (optional)
            # self.T_n1.interpolate(fd.Constant(self.T_0))
            # distortion_solve(self.T_n1, self.T_n)

            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            plt.plot(time_steps, max_temps, 'r-', label='Temperature')
            plt.xlabel('Time [s]')
            plt.ylabel('Max temperature [K]')
            plt.grid(True)
            plt.legend()

            plt.subplot(1, 2, 2)
            plt.plot(time_steps, max_strain_to, 'b-', label='Total Strain')
            plt.plot(time_steps, max_strain_me, 'g--', label='Mechanical Strain')
            plt.plot(time_steps, max_strain_th, 'm:', label='Thermal Strain')
            plt.xlabel('Time [s]')
            plt.ylabel('Max Strain')
            plt.grid(True)
            plt.legend()

            plt.tight_layout()
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
                # end = time.time()
                total_time = time.time() - start_time
                # print("Total time ", f'{(total_time):.3f}')
                log_file.write(f"{self.j:.3e}\t{volume_fraction:.3e}\t{self.c1:3e}\t{self.c2:.3e}\t{self.stressintegral:.3e}\t{total_time:.3e}\n")

        else:
            pass

        return self.j, self.djdrho, self.c, self.dcdrho
