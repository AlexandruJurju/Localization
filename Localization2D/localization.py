import numpy as np
import matplotlib.pyplot as plt

from movement import Movement


class Localization:
    """Performing Bayesian Updating to Produce a Distribution of Likely Positions in the Environment"""

    def __init__(self, world, measurements, movements, prob_hit, prob_miss):
        self.world = world
        self.measurements = measurements
        self.movements = movements
        self.prob_hit = prob_hit
        self.prob_miss = prob_miss

        # initialize the Uniform Prior
        pinit = 1.0 / float(len(world)) / float(len(world[0]))
        self.probabilities = [[pinit for _ in range(len(world[0]))] for _ in range(len(world))]

    def sense(self, prior, world, measurement):
        """
        Compute probabilities after sensing the world (with some confidence).

        Parameters:
            prior (list of lists): Probability distribution.
            world (list of lists): World grid.
            measurement (str): Measurement obtained from the world.

        Returns:
            list of lists: Updated probability distribution after sensing.
        """
        posterior = [[0.0 for _ in range(len(world[0]))] for _ in range(len(world))]

        total_probability = 0.0
        for i in range(len(prior)):
            for j in range(len(prior[i])):
                hit = (measurement == world[i][j])

                # Calculate probability and update posterior matrix
                if hit:
                    posterior[i][j] = prior[i][j] * self.prob_hit
                else:
                    posterior[i][j] = prior[i][j] * self.prob_miss

                total_probability += posterior[i][j]

        # normalize probability values
        for i in range(len(posterior)):
            for j in range(len(prior[0])):
                posterior[i][j] /= total_probability

        return posterior

    # TODO dont use %
    def move(self, prior, motion: [Movement]):
        """
        Compute probabilities after moving through the world (with some confidence).
        Move just shifts the probabilities

        Parameters:
            prior (list of lists): Probability distribution before movement.
            motion (Movement): Movement direction and magnitude.

        Returns:
            list of lists: Updated probability distribution after movement.
        """
        # Initialize new probability matrix
        posterior = [[0.0 for _ in range(len(self.world[0]))] for _ in range(len(self.world))]

        # Iterate through each cell in the probability matrix
        for i in range(len(prior)):
            for j in range(len(prior[0])):
                # Get the x and y delta values for the motion
                motion_value = motion.value

                # Calculate the new position after applying the motion, considering edge wrapping
                new_i = (i - motion_value.x_delta) % len(prior)
                new_j = (j - motion_value.y_delta) % len(prior[i])

                # Update the probability matrix
                posterior[i][j] = prior[new_i][new_j]

        return posterior

    def compute_posterior(self):
        """Call Computation"""
        p = self.probabilities
        for i in range(len(self.measurements)):
            p = self.move(p, self.movements[i])
            # self.visualize_grid(p)
            # self.show(p)

            p = self.sense(p, self.world, self.measurements[i])
            # self.visualize_grid(p)
            self.show(p)

        return p

    @staticmethod
    def show(p):
        """Prints the probability distribution"""
        rows = ['[' + ','.join(map(lambda x: '{0:.5f}'.format(x), r)) + ']' for r in p]
        print('[' + ',\n '.join(rows) + ']')
        print("\n")

    @staticmethod
    def visualize_grid(arr):
        """Visualizes the probability grid"""
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
