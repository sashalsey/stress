import numpy as np # type: ignore
import cyipopt # type: ignore
import time

def CyIpoptWrapper(optimisationClass):
    numberOfVariables = len(optimisationClass.djdrho)
    numberOfConstraints = len(optimisationClass.c)

    # design variable bounds
    lowerBounds = np.ones(numberOfVariables) * optimisationClass.lowerBound
    upperBounds = np.ones(numberOfVariables) * optimisationClass.upperBound

    # constraint bounds
    constraintLowerBounds = np.zeros(numberOfConstraints) #np.array([0, 0])
    constraintUpperBounds = np.zeros(numberOfConstraints) #np.array([0, 0])
    print(constraintLowerBounds)

    # cyipopt class
    class OptimisationSetup:
        def __init__(self):
            pass

        def objective(self, designVariables):
            return optimisationClass.Objective(designVariables)

        def gradient(self, designVariables):
            return optimisationClass.Gradient(designVariables)

        def constraints(self, designVariables):
            return optimisationClass.Constraints(designVariables)

        def jacobian(self, designVariables):
            return optimisationClass.Jacobian(designVariables)

    # cyipopt setup
    optimisationProblem = cyipopt.Problem(
        n=numberOfVariables,
        m=numberOfConstraints,
        problem_obj=OptimisationSetup(),
        lb=lowerBounds,
        ub=upperBounds,
        cl=constraintLowerBounds,
        cu=constraintUpperBounds,)

    optimisationProblem.add_option("linear_solver", "ma97")
    optimisationProblem.add_option("max_iter", optimisationClass.maximumNumberOfIterations)
    optimisationProblem.add_option("accept_after_max_steps", 10)
    optimisationProblem.add_option("hessian_approximation", "limited-memory")
    optimisationProblem.add_option("mu_strategy", "adaptive")
    optimisationProblem.add_option("mu_oracle", "probing")
    optimisationProblem.add_option("tol", 1e-5)
    optimisationProblem.set_problem_scaling(obj_scaling=optimisationClass.gradientScaling)
    optimisationProblem.add_option("nlp_scaling_method", "user-scaling")
    # optimisationProblem.add_option("derivative_test", "second-order")

    # Enable timing statistics
    # optimisationProblem.add_option("timing_statistics") #, "yes")
    optimisationProblem.add_option("print_timing_statistics", "yes")
    # optimisationProblem.print_timing_statistics("yes")

    # Solve the problem
    # start_o = time.time()
    xOptimal, info = optimisationProblem.solve(optimisationClass.designVariables0)
    # end_o = time.time()
    # print("Total time ", f'{(end_o - start_o):.3f}')

    return xOptimal
