from monte_carlo_localization import MonteCarloLocalization
from movement import Direction


def show(p):
    rows = ['[' + ','.join(map(lambda x: '{0:.5f}'.format(x), r)) + ']' for r in p]
    print('[' + ',\n '.join(rows) + ']')


if __name__ == "__main__":
    colours = [['B', 'W', 'W', 'B', 'B'],
               ['B', 'B', 'W', 'B', 'B'],
               ['B', 'B', 'W', 'W', 'B'],
               ['B', 'B', 'B', 'B', 'B'],
               ['B', 'B', 'W', 'W', 'B']]

    start_position = [0, 0]
    movements = [Direction.STAY, Direction.RIGHT, Direction.RIGHT, Direction.RIGHT, Direction.RIGHT]
    measurements = ['B', 'W', 'W', 'B', 'B']

    localization = MonteCarloLocalization(world=colours, measurements=measurements, movements=movements, sensor_prob_correct=0.7, prob_move=0.8)
    posterior = localization.compute_posterior()
    show(posterior)

    # -------------- Instructions ----x

    # The function localize takes the following arguments:
    #
    # colours:
    #        2D list, each entry either 'R' (for red cell) or 'G' (for green cell)
    #
    # measurements:
    #        list of measurements taken by the robot, each entry either 'R' or 'G'
    #
    # motions:
    #        list of actions taken by the robot, each entry of the form [dy,dx],
    #        where dx refers to the change in the x-direction (positive meaning
    #        movement to the right) and dy refers to the change in the y-direction
    #        (positive meaning movement downward)
    #        NOTE: the *first* coordinate is change in y; the *second* coordinate is
    #              change in x
    #
    # sensor_right:
    #        float between 0 and 1, giving the probability that any given
    #        measurement is correct; the probability that the measurement is
    #        incorrect is 1-sensor_right
    #
    # p_move:
    #        float between 0 and 1, giving the probability that any given movement
    #        command takes place; the probability that the movement command fails
    #        (and the robot remains still) is 1-p_move; the robot will NOT overshoot
    #        its destination in this exercise
    #
    # The function should RETURN (not just show or print) a 2D list (of the same
    # dimensions as colours) that gives the probabilities that the robot occupies
    # each cell in the world.
    #
    # Compute the probabilities by assuming the robot initially has a uniform
    # probability of being in any cell.
    #
    # Also assume that at each step, the robot:
    # 1) first makes a movement,
    # 2) then takes a measurement.
    #
    # Motion:
    #  [0,0] - stay
    #  [0,1] - right
    #  [0,-1] - left
    #  [1,0] - down
    #  [-1,0] - up
