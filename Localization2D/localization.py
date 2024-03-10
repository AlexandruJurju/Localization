import numpy as np
import matplotlib.pyplot as plt

from movement import Movement


class Localization:
    def __init__(self, world, measurements, movements, robot_position, prob_hit, prob_miss):
        self.world = world
        self.measurements = measurements
        self.movements = movements
        self.robot_position = robot_position
        self.prob_hit = prob_hit
        self.prob_miss = prob_miss

        # initialize the Uniform Prior
        pinit = 1.0 / float(len(world)) / float(len(world[0]))
        self.probabilities = [[pinit for _ in range(len(world[0]))] for _ in range(len(world))]
        print("Starting probabilities:")
        self.show(self.probabilities)

    def sense(self, old_belief, world, measurement):
        new_belief = [[0.0 for _ in range(len(world[0]))] for _ in range(len(world))]

        sigma = 0.0
        for i in range(len(old_belief)):
            for j in range(len(old_belief[i])):
                hit = (measurement == world[i][j])

                # Calculate probability and update new_belief matrix
                if hit:
                    new_belief[i][j] = old_belief[i][j] * self.prob_hit
                else:
                    new_belief[i][j] = old_belief[i][j] * self.prob_miss

                sigma += new_belief[i][j]

        if sigma > 0:
            # normalize probability values
            for i in range(len(new_belief)):
                for j in range(len(old_belief[0])):
                    new_belief[i][j] /= sigma

        return new_belief

    def move(self, old_belief, movement: Movement):
        # Initialize new probability matrix
        shifted_belief = [[0.0 for _ in range(len(self.world[0]))] for _ in range(len(self.world))]

        motion_value = movement.value

        self.robot_position[0] = self.robot_position[0] + motion_value.x_delta
        self.robot_position[1] = self.robot_position[1] + motion_value.y_delta

        # Iterate through each cell in the probability matrix
        for i in range(len(old_belief)):
            for j in range(len(old_belief[0])):
                # Get the x and y delta values for the motion

                # Calculate the new position after applying the motion, considering edge wrapping
                new_i = (i - motion_value.x_delta) % len(old_belief)
                new_j = (j - motion_value.y_delta) % len(old_belief[i])

                # Update the probability matrix
                shifted_belief[i][j] = old_belief[new_i][new_j]

        return shifted_belief

    def compute_posterior(self):
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
        rows = ['[' + ','.join(map(lambda x: '{0:.5f}'.format(x), r)) + ']' for r in p]
        print('[' + ',\n '.join(rows) + ']')
        print("\n")

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
