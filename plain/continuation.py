from cyIpoptSetup import CyIpoptWrapper
from optimise import OptimisationLoop
from plot import plot
import time
start = time.time()

# flake8 initialisation bug
rhoOptimal = None
gradientScaling = None
constraintScaling = None
jacobianScaling = None

outputFolder2 = ("results/")
resultsFile = open(outputFolder2 + "combined_iteration_results.txt", "w")
resultsFile.write("Obj\tVolume Fraction\tMax Stress\tC1\tC2\tStressIntegral\n")
resultsFile.close()

for i in range(1):

    if i == 0: ##### Compliance Objective; Volume constraint #####
        optimisationClass = OptimisationLoop(1, 0) 
        optimisationClass.maximumNumberOfIterations = 18
        optimisationClass.pnorm = 8
    elif i == 1:##### Compliance Objective; Volume and p-norm stress constraint #####
        optimisationClass = OptimisationLoop(2, stressintegral)
        optimisationClass.maximumNumberOfIterations = 50
        optimisationClass.pnorm = 8
        optimisationClass.initial_stressintegral = stressintegral
    # elif i == 2:##### Compliance Objective; Volume and p-norm stress constraint #####
    #     optimisationClass = OptimisationLoop(3, stressintegral)
    #     optimisationClass.maximumNumberOfIterations = 100
    #     optimisationClass.pnorm = 8
    #     optimisationClass.initial_stressintegral = stressintegral

    optimisationClass.beta = 4

    if i == 0:
        pass # first iteration
    else:
        # initialise design variables with previous solution
        optimisationClass.variableInitialisation = True
        optimisationClass.rho0 = rhoOptimal

        # initialise scaling with initial values (i = 0 scaling)
        optimisationClass.scalingInitialisation = True
        optimisationClass.gradientScaling = gradientScaling
        optimisationClass.constraintScaling = constraintScaling
        optimisationClass.jacobianScaling = jacobianScaling

    # optimisationClass = OptimisationLoop(1, 0) 
    # optimisationClass.maximumNumberOfIterations = 30
    # optimisationClass.pnorm = 8
    # optimisationClass.beta = 4

    # optimisationClass.is_first_iteration = True
    # optimisationClass.is_last_iteration = False

    # optimisation setup
    optimisationClass.OptimisationSetup()

    # execute optimisation procedure using ipopt
    rhoOptimal = CyIpoptWrapper(optimisationClass)

    # optimisationClass.is_first_iteration = False
    # optimisationClass.is_last_iteration = True
    # _ = optimisationClass.ForwardSolve.Solve(rhoOptimal)

    # output scaling
    gradientScaling = optimisationClass.gradientScaling
    constraintScaling = optimisationClass.constraintScaling
    jacobianScaling = optimisationClass.jacobianScaling

    end = time.time()
    print("Total time ", f'{(end - start):.3f}')
    plot_results = plot()
    # plot_time_per_iteration = plot()
    stressintegral = plot_results.get_last_stress_integral() 

