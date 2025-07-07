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
        self.t_heat = 1.0 # time between deposit and next layer
        self.dt = 0.05 # timestep for t_heat
        
        # pseudo-density functional
        self.beta = beta  # heaviside projection paramesteepnesster
        self.eta0 = 0.5  # midpoint of projection filter
        
        self.Vlimit = 0.99 # volume fraction constraint
        self.pnorm = pnorm # p-norm stress penalty
        self.initial_stressintegral = float(initial_stressintegral)

    def GenerateMesh(self,):
        self.mesh = fd.RectangleMesh(self.nx, self.ny, self.lx, self.ly, quadrilateral=True)
        self.gradientScale = (self.nx * self.ny) / (self.lx * self.ly)
        self.dx = fd.Measure('dx', domain=self.mesh)
        self.ds = fd.ds(domain=self.mesh)
        self.n = fd.FacetNormal(self.mesh)

    def Setup(self):
        self.nx, self.ny = 10, 200
        self.lx, self.ly =  0.05, 1.0
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
        self.mech_bc1 = fd.DirichletBC(self.displace, fd.Constant((0, 0)), 3) 
        # self.mech_bc2 = fd.DirichletBC(self.displace, fd.Constant((0, 0)), 4)
        # self.mech_bc3 = fd.DirichletBC(self.displace, fd.Constant((0, 0)), 2) 
        # self.mech_bc4 = fd.DirichletBC(self.displace, fd.Constant((0, 0)), 1)
        # self.mech_bc3 = FixedDirichletBC(self.displace.sub(0), fd.Constant(0), np.array([0.025, 0.0]))
        self.temp_bc1 = fd.DirichletBC(self.templace, fd.Constant(0.0), 3)

        # output files
        self.rho_hatFile = VTKFile(self.outputFolder + "rho_hat.pvd")
        self.rho_hatFunction =fd.Function(self.densityspace, name="rho_hat")

        self.k_rho = fd.Function(self.densityspace, name="k_rho") # density-dependent thermal conductivity
        self.cp = fd.Function(self.densityspace, name="cp") # density-dependent thermal storage

        self.tempFile = VTKFile(self.outputFolder + "temp.pvd")
        self.T_n = fd.Function(self.templace, name="Temperature")
        self.T_n1 = fd.Function(self.templace)

        self.strain_therm = fd.Function(self.DG0, name="strain_therm")
        self.strain_thermFile = VTKFile(self.outputFolder + "strain_therm.pvd")

        self.strain_th_increment = fd.Function(self.DG0, name="strain_th_increment")
        self.strain_th_incrementFile = VTKFile(self.outputFolder + "strain_th_increment.pvd")

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
            self.rho_hatFile.write(self.rho_hatFunction)
            
            ##### Temperature ######
            w = fd.TestFunction(self.templace)
            
            Id = fd.Identity(self.mesh.geometric_dimension())
            # self.E = self.E_0 + (self.E_1 - self.E_0) * (self.rho_hat ** self.penalisationExponent)          
            # self.strain_therm.interpolate(fd.Constant(0))
            self.k_rho.interpolate(self.k_1)
            self.cp.interpolate(self.cp_1)
            self.T_n.interpolate(fd.Constant(self.T_0))
            self.tempFile.write(self.T_n)
            self.T_n1.interpolate(fd.Constant(self.T_1))
            
            # Functions
            # mixed_functionspace = self.displace * self.templace
            # (u_t, T_t) = fd.TrialFunctions(mixed_functionspace)
            # (v_t, q_t) = fd.TestFunctions(mixed_functionspace)

            # thermal_strain = self.alpha_1 * (self.T_n1 - self.T_n) * self.rho_hat * Id

            # self.strain_th_increment.interpolate(fd.project(thermal_strain, self.DG0))
            # self.strain_therm.assign(self.strain_therm + self.strain_th_increment)

            # self.strain_therm.interpolate(self.strain_th_increment + self.strain_therm) # Thermal strain
            # self.strain_thermFile.write(self.strain_therm)     
            
            # def strain_total(u): return 0.5 * (fd.grad(u) + fd.grad(u).T) # Total strain 

            # # Weak form
            # stress_mech = E * (strain_total(u_t) - self.strain_therm)
            # a_t = fd.inner(stress_mech, strain_total(v_t))

            # # Solve
            # sol = fd.Function(mixed_functionspace)
            # fd.solve(a_t == 0, sol, bcs=[self.mech_bc1, self.temp_bc1])
            # u_t, T_t = sol.split()
            # self.u_tFunction.assign(u_t)
            # self.u_tFile.write(self.u_tFunction)

            # self.T_n.assign(self.T_n1)
            # self.tempFile.write(self.T_n)

            # self.stress.assign(fd.project(stress(self.E, u_t),self.DG0))
            # self.stressFile.write(self.stress)
            # print("Got to here")
            # sys.exit()

            current_time = 0.0
            # max_temps = []
            # time_steps = []
            # max_stress = []
            # max_temps.append(np.max(self.T_n.vector().get_local()))
            # time_steps.append(current_time)
            # max_stress.append(np.max(self.stress.vector().get_local()))

            while current_time < self.t_heat:
                #### Temperature Solve ####
                w = fd.TestFunction(self.templace)
                F = (self.cp * (self.T_n1 - self.T_n) * w * self.dx + self.dt * self.k_rho * fd.dot(fd.grad(self.T_n1), fd.grad(w)) * self.dx)
                F += self.dt * self.h * (self.T_n1 - self.T_0) * w * self.ds(1)
                F += self.dt * self.h * (self.T_n1 - self.T_0) * w * self.ds(2)
                F += self.dt * self.h * (self.T_n1 - self.T_0) * w * self.ds(4)
                fd.solve(F == 0, self.T_n1, bcs=[self.temp_bc1])

                #### Distortion Solve ####
                mixed_functionspace = self.displace * self.templace
                (u_t, T_t) = fd.TrialFunctions(mixed_functionspace)
                (v_t, q_t) = fd.TestFunctions(mixed_functionspace)

                thermal_strain = self.alpha_1 * (self.T_n1 - self.T_n) * self.rho_hat * Id

                self.strain_th_increment.interpolate(fd.project(thermal_strain, self.DG0))
                self.strain_therm.assign(self.strain_therm + self.strain_th_increment)

                self.strain_therm.interpolate(self.strain_th_increment + self.strain_therm) # Thermal strain
                self.strain_thermFile.write(self.strain_therm)     
                
                def strain_total(u): return 0.5 * (fd.grad(u) + fd.grad(u).T) # Total strain 

                # Weak form
                self.E = self.E_0 + (self.E_1 - self.E_0) * (self.rho_hat ** self.penalisationExponent)  
                a_t = fd.inner(v_t, fd.div(self.E * strain_total(u_t))) * fd.dx # Divergence(Vector)
                L_t = fd.inner(q_t, fd.grad(self.E * self.strain_therm)) * fd.dx # Gradient(Scalar)

                # Solve
                sol = fd.Function(mixed_functionspace)
                fd.solve(a_t == L_t, sol, bcs=[self.mech_bc1, self.temp_bc1])
                u_t, T_t = sol.split()
                self.u_tFunction.assign(u_t)
                self.u_tFile.write(self.u_tFunction)

                self.T_n.assign(self.T_n1)
                self.tempFile.write(self.T_n)

                # self.stress.assign(fd.project(stress(self.E, u_t),self.DG0))
                # self.stressFile.write(self.stress)
                print("Got to here")
                sys.exit()

                max_temps.append(np.max(self.T_n.vector().get_local()))
                time_steps.append(current_time)
                max_stress.append(np.max(self.stress.vector().get_local()))
                        
                current_time += self.dt

            ##### Plot temperature over transient loop #####
            plt.figure(figsize=(10, 6))
            plt.plot(time_steps, max_temps, 'k-', marker='o', markersize=5, label='Max Temperature')
            plt.xlabel('Time (s)')
            plt.ylabel('Temperature (K)')
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout()
            plt.show(block=False)
            plt.pause(10)
            plt.close()

            plt.figure(figsize=(10, 6))
            plt.plot(time_steps, max_stress, 'k-', marker='o', markersize=5, label='Max Temperature')
            plt.xlabel('Time (s)')
            plt.ylabel('Stress (Pa)')
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout()
            plt.show(block=False)
            plt.pause(10)
            plt.close()
            
            ###### Final Thermal Stress ######
            # sigma_xx = fd.project(sigma(u_t)[0, 0], self.DG0)
            # sigma_yy = fd.project(sigma(u_t)[1, 1], self.DG0)
            # sigma_xy = fd.project(sigma(u_t)[0, 1], self.DG0)
            # self.sigma_xx_f.assign(sigma_xx)
            # self.sigma_yy_f.assign(sigma_yy)
            # self.sigma_xy_f.assign(sigma_xy)
            # self.sigma_xx_File.write(self.sigma_xx_f)
            # self.sigma_yy_File.write(self.sigma_yy_f)
            # self.sigma_xy_File.write(self.sigma_xy_f)
            # von_mises_stress = fd.sqrt(sigma_xx**2 - sigma_xx * sigma_yy + sigma_yy**2 + 3 * sigma_xy**2)
            # von_mises_total_proj = fd.project(von_mises_stress, self.DG0)
            # # print("max_stress", np.max(von_mises_total_proj.vector().get_local()))
            # self.thermalstress.assign(von_mises_total_proj)
            # self.thermalstressFile.write(self.thermalstress)
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

            # sigma_xx = fd.project(sigma(u)[0, 0], self.DG0)
            # sigma_yy = fd.project(sigma(u)[1, 1], self.DG0)
            # sigma_xy = fd.project(sigma(u)[0, 1], self.DG0)
            # von_mises_stress = fd.sqrt(sigma_xx**2 - sigma_xx * sigma_yy + sigma_yy**2 + 3 * sigma_xy**2)
            # von_mises_total_proj = fd.project(von_mises_stress, self.DG0)
            # # max_stress = np.max(von_mises_total_proj.vector().get_local())
            # print("max_stress", np.max(von_mises_total_proj.vector().get_local()))
            # self.strainmechFunction.assign(von_mises_total_proj)
            # self.strainmechFile.write(self.strainmechFunction)

            # self.stressintegral = fd.assemble(((von_mises_total_proj ** self.pnorm) * self.dx) ) ** (1/self.pnorm)
            # print("Stress integral = ", self.stressintegral)
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

            # elif self.scenario == 2:
            #     ##### Objective: Stress #####
            #     self.j = self.stressintegral #fd.assemble(fd.inner(T, u) * self.ds(4))

            #     ##### Constraints #####
            #     volume_fraction = (1 / self.meshVolume) * fd.assemble(self.rho_hat * self.dx)
            #     self.c1 = self.Vlimit - volume_fraction
            #     self.c2 = 8e3 - fd.assemble(fd.inner(T, u) * self.ds(4))

            #     ##### Objective sensitivities #####
            #     self.djdrho = (fda.compute_gradient(self.j, fda.Control(self.rho)).vector().get_local()) / self.gradientScale

            #     ##### Constraint sensitivities #####
            #     self.dc1drho = (fda.compute_gradient(self.c1, fda.Control(self.rho)).vector().get_local()) / self.gradientScale
            #     self.dc2drho = (fda.compute_gradient(self.c2, fda.Control(self.rho)).vector().get_local()) / self.gradientScale

            #     self.c = np.array([self.c1, self.c2])
            #     self.dcdrho = np.concatenate((self.dc1drho, self.dc2drho))

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
