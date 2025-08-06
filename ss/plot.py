import numpy as np # type: ignore
import matplotlib.pyplot as plt # type:ignore

class plot:
    def __init__(self):
        self.file_path = "results/combined_iteration_results.txt"
        self.load_data()
        self.plot_results()

    def load_data(self):
        data = np.loadtxt(self.file_path, skiprows=1)
        self.objective = data[:, 0]  # Objective (X-axis)
        self.volume_fraction = data[:, 1]  # Volume fraction (Y-axis)
        self.c1 = data[:, 2]  # Constraint 1 (Y-axis)
        self.c2 = data[:, 3]  # Constraint 2 (Y-axis)
        # self.max_stress = data[:, 4]  # Max Temp (Y-axis)
        self.stressintegral = data[:, 4]
        self.times = data[:,5]
        self.iterations = np.arange(1, len(self.objective) + 1)
        print("Vol frac, Obj, c1, c2, max stress")
        print(self.volume_fraction[-1], " ", self.objective[-1], " ", self.c1[-1], " ", self.c2[-1] )#, " ", self.max_stress[-1])
        # print(self.max_stress[-1])
        # print(self.stressintegral[-1])
    
    def get_last_stress_integral(self):
        return self.stressintegral[-1]

    def plot_results(self):
        fig, ax1 = plt.subplots(figsize=(18, 6))

        # Plot volume fraction on the left y-axis
        ln1 = ax1.plot(self.iterations, self.volume_fraction, 'k-', label="Volume Fraction")
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("Volume Fraction", color='k')
        ax1.tick_params(axis='y', labelcolor='k')
        ax1.grid(True, linestyle='--', linewidth=0.5)

        ax2 = ax1.twinx()
        ln2 = ax2.plot(self.iterations, self.objective, 'g-', label="Objective")
        ax2.set_ylabel("Objective", color='g')
        ax2.tick_params(axis='y', labelcolor='g')

        ax3 = ax1.twinx()
        ln3 = ax3.plot(self.iterations, self.c1, 'r-', label="Constraint 1")
        ax3.set_ylabel("Constraint 1", color='r')
        ax3.tick_params(axis='y', labelcolor='r')

        ax4 = ax1.twinx()
        ln4 = ax4.plot(self.iterations, self.c2, 'm-', label="Constraint 2")
        ax4.set_ylabel("Constraint 2", color='m')
        ax4.tick_params(axis='y', labelcolor='m')

        # ax5 = ax1.twinx()
        # ln5 = ax5.plot(self.iterations, self.max_stress, 'b-', label="Max Temp")
        # ax5.set_ylabel("Max Stress", color='b')
        # ax5.tick_params(axis='y', labelcolor='b')

        lns = ln1 + ln2 + ln3 + ln4  #+ ln5
        labels = [l.get_label() for l in lns]
        ax1.legend(lns, labels, loc="upper left")
        plt.show()      
        # Plot time
        # plt.figure(figsize=(6, 4))         
        # plt.plot(self.iterations, self.times)
        # plt.xlabel("Iteration")
        # plt.ylabel("Time (s)")
        # plt.grid(True, linestyle='--', linewidth=0.5)
        # plt.show()