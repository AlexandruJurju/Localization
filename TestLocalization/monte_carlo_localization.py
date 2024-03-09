import numpy as np
import matplotlib.pyplot as plt

from movement import Direction


class MonteCarloLocalization():
    """Performing Bayesian Updating to Produce a Distribution of Likely Positions in the Environment"""

    def __init__(self, world, measurements, movements, sensor_prob_correct, prob_move):
        self.world = world
        self.measurements = measurements
        self.motions = movements
        self.sensor_prob_correct = sensor_prob_correct
        self.prob_move = prob_move

        # initialize the Uniform Prior
        pinit = 1.0 / float(len(world)) / float(len(world[0]))
        self.p = [[pinit for _ in range(len(world[0]))] for _ in range(len(world))]
        self.visualize_grid(self.p)

    def sense(self, p, world, measurement):
        """Compute probabilities after sensing the world (with some confidence)"""
        q = [[0.0 for _ in range(len(world[0]))] for _ in range(len(world))]

        s = 0.0
        for i in range(len(p)):
            for j in range(len(p[i])):
                hit = (measurement == world[i][j])
                q[i][j] = p[i][j] * (hit * self.sensor_prob_correct + (1 - hit) * (1 - self.sensor_prob_correct))
                s += q[i][j]

        # normalize
        for i in range(len(q)):
            for j in range(len(p[0])):
                q[i][j] /= s

        return q

    def move(self, p, motion: [Direction]):
        """Compute probabilities after moving through world (with some confidence)"""
        q = [[0.0 for _ in range(len(self.world[0]))] for _ in range(len(self.world))]

        for i in range(len(p)):
            for j in range(len(p[0])):
                motion_value = motion.value
                q[i][j] = (self.prob_move * p[(i - motion_value.x_delta) % len(p)][(j - motion_value.y_delta) % len(p[i])]) + ((1 - self.prob_move) * p[i][j])
        return q

    def compute_posterior(self):
        """Call Computation"""
        p = self.p
        for i in range(len(self.measurements)):
            p = self.move(p, self.motions[i])
            self.visualize_grid(p)
            p = self.sense(p, self.world, self.measurements[i])
            self.visualize_grid(p)

        return p

    @staticmethod
    def visualize_grid(arr):
        grid = np.array(arr)
        fig, ax = plt.subplots()

        single_color_grid = np.zeros_like(grid)
        ax.imshow(single_color_grid, cmap='Greys', interpolation='none')

        for i in range(len(arr)):
            for j in range(len(arr[0])):
                # Add the text for the probability at i,j
                ax.text(j, i, round(arr[i][j], 2), ha='center', va='center', color='black')

                # Add borders
                rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, edgecolor='black', linewidth=1, fill=False)
                ax.add_patch(rect)

        ax.set_xticks(np.arange(-0.5, len(arr[0]), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(arr), 1), minor=True)
        ax.grid(which='minor', color='black', linestyle='-', linewidth=1)

        # Add row numbers
        for i in range(len(arr)):
            ax.text(-1, i, str(i), ha='center', va='center', color='black')

        # Add column numbers
        for j in range(len(arr[0])):
            ax.text(j, -1, str(j), ha='center', va='center', color='black')

        ax.set_yticklabels([])

        plt.show()
