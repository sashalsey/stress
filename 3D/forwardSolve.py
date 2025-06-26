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
        self.yieldstress = 2.5e8
        self.nu = 0.3

        # thermal properties
        self.T_0, self.T_1 = 300.0, 2000.0 # atmospheric temperature, heating temperature
        self.k_0, self.k_1 = 0.026, 17.0 # thermal conduction coeff of air, titanium
        self.alpha_0, self.alpha_1 = 2e-3, 8e-6 # linear thermal expansion coeff of air, titanium
        self.h = 100.0 # thermal convection coeff
        self.cp_0, self.cp_1 = 1e3, 500 # coeff specific heat capacity of air, titanium
        self.rho_material = 4500.0
        self.t_heat = 2.0 # time between deposit and next layer
        self.dt = 1.0 # timestep

        # Baseplate properties
        self.E_b = 1.9e11
        self.k_b = 15.0
        self.alpha_b = 16e-6
        self.cp_b = 500
        self.rho_b = 8000.0
        
        # pseudo-density functional
        self.beta = beta  # heaviside projection paramesteepnesster
        self.eta0 = 0.5  # midpoint of projection filter
        
        self.Vlimit = 0.99 # volume fraction constraint
        self.pnorm = pnorm # p-norm stress penalty
        self.initial_stressintegral = float(initial_stressintegral)

    def GenerateMesh(self,):
        self.mesh = fd.BoxMesh(self.nx, self.ny, self.nz, self.lx, self.ly, self.lz, hexahedral=False, diagonal='default')
        self.gradientScale = (self.nx * self.ny * self.nz) / (self.lx * self.ly * self.lz)
        self.dx = fd.Measure('dx', domain=self.mesh)
        self.ds = fd.ds(domain=self.mesh)
        self.n = fd.FacetNormal(self.mesh)
        print("Mesh generated")

    def Setup(self):
        self.nx, self.ny, self.nz = 10, 20, 4
        self.lx, self.ly, self.lz =  0.05, 0.1, 0.02
        self.cellsize = self.lx / self.nx
        self.GenerateMesh()
        self.meshVolume = fd.assemble(1 * fd.dx(domain=self.mesh))
                                   
        # function spaces
        self.densityspace = fd.FunctionSpace(self.mesh, "CG", 1)
        self.displace = fd.VectorFunctionSpace(self.mesh, "CG", 2)
        self.templace = fd.FunctionSpace(self.mesh, "CG", 1)
        self.cooling_rate = fd.FunctionSpace(self.mesh, "CG", 1)
        self.DG0 = fd.FunctionSpace(self.mesh, "CG", 1)

        self.rho = fd.Function(self.densityspace)

        # number of nodes in mesh
        self.numberOfNodes = len(self.rho.vector().get_local())

        # compute helmholtz filter radius
        self.helmholtzFilterRadius = 1 * (self.meshVolume / self.mesh.num_cells()) ** (1 / self.mesh.cell_dimension())

        # boundary conditions
        # 1: vertical y lhs, 2: vertical y rhs, 3: horizontal x bottom, 4: horizontal x top
        # BCs for compliance optimisation
        self.mech_bc1 = fd.DirichletBC(self.displace, fd.Constant((0, 0, 0)), 3) # fixed bottom face
        # self.mech_bc2 = FixedDirichletBC(self.displace.sub(0), fd.Constant(0), np.array([0.05, 0.1, 0.025])) # pinned top rhs corner
        # BCs for thermal
        self.mech_bc3 = FixedDirichletBC(self.displace.sub(0), fd.Constant(0), np.array([0.025, 0.0, 0.01]))
        # self.mech_bc3 = fd.DirichletBC(self.displace, fd.Constant((0, 0, 0)), 3) # fixed bottom face
        self.temp_bc1 = fd.DirichletBC(self.templace, fd.Constant(300.0), 3)

        # output files
        self.rho_hatFile = VTKFile(self.outputFolder + "rho_hat.pvd")
        self.rho_hatFunction =fd.Function(self.densityspace, name="rho_hat")

        self.rho_dep =fd.Function(self.densityspace, name="rho_dep") # deposited rho
        self.rho_depFile = VTKFile(self.outputFolder + "rho_dep.pvd")
        self.k_rho = fd.Function(self.densityspace, name="k_rho") # density-dependent thermal conductivity
        self.cp = fd.Function(self.densityspace, name="cp") # density-dependent thermal storage

        self.tempFile = VTKFile(self.outputFolder + "temp.pvd")
        self.T_n = fd.Function(self.templace, name="Temperature")

        self.thermalstrain = fd.Function(self.densityspace, name="thermalstrain")
        self.thermalstrainFile = VTKFile(self.outputFolder + "thermalstrain.pvd")
        self.thermalstress = fd.Function(self.densityspace, name="thermalstress")
        self.thermalstressFile = VTKFile(self.outputFolder + "thermalstress.pvd")
        self.strainincrement = fd.Function(self.densityspace, name="strain_increment")
        self.strainincrementFile = VTKFile(self.outputFolder + "strain_increment.pvd")
        self.residualstrain = fd.Function(self.densityspace, name="residual_thermal_strain")
        self.residualstrainFile = VTKFile(self.outputFolder + "residualthermalstrain.pvd")
    
        # mechanical distortion displacement
        self.uFile = VTKFile(self.outputFolder + "u.pvd")
        self.uFunction = fd.Function(self.displace, name="u")

        # thermal distortion displacement
        self.u_tFile = VTKFile(self.outputFolder + "u_t.pvd")
        self.u_tFunction = fd.Function(self.displace, name="u_t")

        self.Efile = VTKFile(self.outputFolder + "E.pvd")

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

            x, y, z = fd.SpatialCoordinate(self.mesh)

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
            
            ##### Transient Temperature Loop ######
            n = 10 # number of layers
            layer_height = self.ly / n  # building in y direction

            # Initialise temperature field
            self.T_n1 = fd.Function(self.templace)
            w = fd.TestFunction(self.templace)
            
            self.T_n.interpolate(fd.Constant(self.T_0))
            self.E = fd.Function(self.densityspace)

            plastic = -0.01 #self.yieldstress / self.E_1
            self.thermalstrain.interpolate(fd.Constant(0))

            Id = fd.Identity(self.mesh.geometric_dimension())
            def epsilon(u): return 0.5 * (fd.grad(u) + fd.grad(u).T)
            def sigma(u): return lambda_ * fd.div(u) * Id + 2 * mu * epsilon(u)
            
            for i in range(n):
                print("i = ", i)
                layer_height_dep = layer_height * (i + 1)
                k = 1000 # steepness for hyperbolid tangent like beta, but steeper

                if i == 0: # Baseplate layer 0
                    self.rho_dep.interpolate(0.5 * (1 + fd.tanh(k * ( - y + layer_height_dep))) * self.rho_hat)
                    # self.rho_depFile.write(self.rho_dep)
                    self.k_rho.interpolate(self.k_0 + (self.k_b - self.k_0) * (self.rho_dep ** self.penalisationExponent))
                    self.cp.interpolate(1.2 * self.cp_0 + (self.rho_b * self.cp_b - 1.2 * self.cp_0) * (self.rho_dep ** self.penalisationExponent))
                    self.T_n.interpolate(self.T_n * (1 + 0.5 * (fd.tanh(k * ( - y + layer_height_dep -layer_height)) + fd.tanh(k * (y - layer_height_dep))) * 0.5 * (1 + fd.tanh(k * (self.rho_hat - 0.5)))) 
                                        + self.T_n * (-0.5*(fd.tanh(k*(-y+layer_height_dep-layer_height))+fd.tanh(k*(y-layer_height_dep))))*(0.5*(1+fd.tanh(k*(self.rho_hat - 0.5)))))

                    self.tempFile.write(self.T_n)
                
                elif i >= 1 and i <= n-1: # Build layers
                    self.rho_dep.interpolate(0.5 * (1 + fd.tanh(k * ( - y + layer_height_dep))) * self.rho_hat)
                    # self.rho_depFile.write(self.rho_dep)
                    self.k_rho.interpolate(self.k_0 + (self.k_1 - self.k_0) * (self.rho_dep ** self.penalisationExponent))
                    self.cp.interpolate(1.2 * self.cp_0 + (self.rho_material * self.cp_1 - 1.2 * self.cp_0) * (self.rho_dep ** self.penalisationExponent))
                    self.T_n.interpolate(self.T_n * (1 + 0.5 * (fd.tanh(k * ( - y + layer_height_dep -layer_height)) + fd.tanh(k * (y - layer_height_dep))) * 0.5 * (1 + fd.tanh(k * (self.rho_hat - 0.5)))) 
                                        + self.T_1 * (-0.5*(fd.tanh(k*(-y+layer_height_dep-layer_height))+fd.tanh(k*(y-layer_height_dep))))*(0.5*(1+fd.tanh(k*(self.rho_hat - 0.5)))))

                    self.tempFile.write(self.T_n)
                    
                    current_time = 0.0 
                    while current_time < self.t_heat:
                        F = (self.cp * (self.T_n1 - self.T_n) * w * self.dx + self.dt * self.k_rho * fd.dot(fd.grad(self.T_n1), fd.grad(w)) * self.dx)
                        F += self.dt * self.h * (self.T_n1 - self.T_0) * w * self.ds(1)
                        F += self.dt * self.h * (self.T_n1 - self.T_0) * w * self.ds(2)
                        F += self.dt * self.h * (self.T_n1 - self.T_0) * w * self.ds(4)
                        fd.solve(F == 0, self.T_n1, bcs=[self.temp_bc1])

                        self.strainincrement.interpolate(self.alpha_1 * (self.T_n1 - self.T_1) * self.rho_dep)
                        self.strainincrement.interpolate(0.5 * (1 + fd.tanh(k * ( - y + layer_height_dep))) * self.strainincrement)
                        # self.strainincrementFile.write(self.strainincrement)
                        self.thermalstrain.interpolate(self.strainincrement + self.thermalstrain)
                        # self.thermalstrainFile.write(self.thermalstrain)
                        self.residualstrain.interpolate(0.5 * (- 1 + fd.tanh(k * (self.thermalstrain - plastic))) * -1 * self.thermalstrain)
                        self.residualstrainFile.write(self.residualstrain)
                        
                        self.E.interpolate(self.E_0 + 
                        (self.E_1 - self.E_0) * (self.rho_dep ** self.penalisationExponent) * 0.5 * (1 + fd.tanh(k * (y - layer_height))) * 0.5 * (1 + fd.tanh(k * (-y + layer_height_dep))) +
                        (self.E_b - self.E_0) * (self.rho_dep ** self.penalisationExponent) * 0.5 * (1 + fd.tanh(k * (-y + layer_height))))
                        self.Efile.write(self.E)
                        # self.E = self.E_0 + (self.E_1 - self.E_0) * (self.rho_hat ** self.penalisationExponent)
                        lambda_ = (self.E * self.nu) / ((1 + self.nu) * (1 - 2 * self.nu))
                        mu = (self.E) / (2 * (1 + self.nu))

                        u_t = fd.TrialFunction(self.displace)
                        v_t = fd.TestFunction(self.displace)

                        # Weak form
                        a_t = fd.inner(sigma(u_t), epsilon(v_t)) * fd.dx # Mechanical
                        L_t = fd.inner(sigma(v_t), self.thermalstrain * Id) * fd.dx

                        # Solve
                        u_t = fd.Function(self.displace)
                        fd.solve(a_t == L_t, u_t, bcs=self.mech_bc3)
                        self.u_tFunction.assign(u_t)
                        self.u_tFile.write(self.u_tFunction)
                        print(f"Max abs u_t {np.max(np.abs(u_t.vector().get_local())):.3g}")
                        
                        self.T_n.assign(self.T_n1)
                        self.tempFile.write(self.T_n)
                        
                        current_time += self.dt
                
                if i == n - 1: # Forced cooled to ambient
                    self.T_n1.interpolate(self.T_0)
                    self.T_n.assign(self.T_n1)
                    self.tempFile.write(self.T_n)

                    self.strainincrement.interpolate(self.alpha_1 * (self.T_n1 - self.T_1) * self.rho_dep)
                    self.strainincrement.interpolate(0.5 * (1 + fd.tanh(k * ( - y + layer_height_dep))) * self.strainincrement)
                    # self.strainincrementFile.write(self.strainincrement)
                    self.thermalstrain.interpolate(self.strainincrement + self.thermalstrain)
                    # self.thermalstrainFile.write(self.thermalstrain)
                    self.residualstrain.interpolate(0.5 * (- 1 + fd.tanh(k * (self.thermalstrain - plastic))) * -1 * self.thermalstrain)
                    self.residualstrainFile.write(self.residualstrain)

                    self.E.interpolate(self.E_0 + 
                        (self.E_1 - self.E_0) * (self.rho_dep ** self.penalisationExponent) * 0.5 * (1 + fd.tanh(k * (y - layer_height))) * 0.5 * (1 + fd.tanh(k * (-y + layer_height_dep))) +
                        (self.E_b - self.E_0) * (self.rho_dep ** self.penalisationExponent) * 0.5 * (1 + fd.tanh(k * (-y + layer_height))))
                    self.Efile.write(self.E)
                    lambda_ = (self.E * self.nu) / ((1 + self.nu) * (1 - 2 * self.nu))
                    mu = (self.E) / (2 * (1 + self.nu))

                    u_t = fd.TrialFunction(self.displace)
                    v_t = fd.TestFunction(self.displace)

                    # Weak form
                    a_t = fd.inner(sigma(u_t), epsilon(v_t)) * fd.dx # Mechanical
                    L_t = fd.inner(sigma(v_t), self.thermalstrain * Id) * fd.dx

                    # Solve
                    u_t = fd.Function(self.displace)
                    fd.solve(a_t == L_t, u_t, bcs=self.mech_bc3)
                    self.u_tFunction.assign(u_t)
                    self.u_tFile.write(self.u_tFunction)
                    print(f"Max abs u_t coooooooling {np.max(np.abs(u_t.vector().get_local())):.3g}")

                        # max_temps.append(np.max(self.T_n.vector().get_local()))
                        # count += self.dt2
                        # time_steps.append(count)
            # self.stressintegral = fd.assemble(((self.strainincrement ** self.pnorm) * self.dx) ) ** (1/self.pnorm)
            sys.exit()  
            
            
            ##### Plot temperature over transient loop #####
            # plt.figure(figsize=(10, 6))
            # plt.plot(time_steps, max_temps, 'k-', marker='o', markersize=5, label='Max Temperature')
            # plt.xlabel('Time (s)')
            # plt.ylabel('Temperature (K)')
            # plt.grid(True, linestyle='--', alpha=0.6)
            # plt.tight_layout()
            # plt.show(block=False)
            # plt.pause(2)
            # plt.close()
            
            ###### Final Thermal Stress ######
            # sigma_xx = fd.project(sigma(u_t)[0, 0], self.DG0)
            # sigma_yy = fd.project(sigma(u_t)[1, 1], self.DG0)
            # sigma_xy = fd.project(sigma(u_t)[0, 1], self.DG0)
            # von_mises_stress = fd.sqrt(sigma_xx**2 - sigma_xx * sigma_yy + sigma_yy**2 + 3 * sigma_xy**2)
            # von_mises_total_proj = fd.project(von_mises_stress, self.DG0)
            # # print("max_stress", np.max(von_mises_total_proj.vector().get_local()))
            # self.thermalstress.assign(von_mises_total_proj)
            # self.thermalstressFile.write(self.thermalstress)

            # Stress Calculation
            # sigma_xx = fd.project(sigma(u)[0, 0], self.DG0)
            # sigma_yy = fd.project(sigma(u)[1, 1], self.DG0)
            # sigma_zz = fd.project(sigma(u)[2, 2], self.DG0)
            # sigma_xy = fd.project(sigma(u)[0, 1], self.DG0)
            # sigma_yz = fd.project(sigma(u)[1, 2], self.DG0)
            # sigma_zx = fd.project(sigma(u)[2, 0], self.DG0)

            print("ctrl z")
            time.sleep(5.0)

            self.E = self.E_0 + (self.E_1 - self.E_0) * (self.rho_hat ** self.penalisationExponent)
            lambda_ = (self.E * self.nu) / ((1 + self.nu) * (1 - 2 * self.nu))
            mu = (self.E) / (2 * (1 + self.nu))

            ###### Mechanical Load Compliance ######
            u = fd.TrialFunction(self.displace)
            v = fd.TestFunction(self.displace)

            # Surface traction
            T = fd.conditional(
                fd.And(fd.gt(y, 0.095), fd.gt(x, 0.04)),
                fd.as_vector([-3e8, 0, 0]),
                fd.as_vector([0, 0, 0]))

            # Weak form
            a = fd.inner(sigma(u), epsilon(v)) * fd.dx
            L = fd.dot(T, v) * fd.ds(6)

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
