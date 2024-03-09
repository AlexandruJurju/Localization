from markov_localization import MonteCarloLocalization
from movement import Movement

if __name__ == "__main__":
    colours = [['B', 'W', 'B', 'B', 'B'],
               ['B', 'W', 'B', 'B', 'B'],
               ['B', 'B', 'B', 'B', 'B'],
               ['B', 'B', 'B', 'B', 'B'],
               ['B', 'B', 'B', 'B', 'B']]

    start_position = [0, 0]
    movements = [Movement.STAY, Movement.RIGHT, Movement.RIGHT]
    measurements = ['B', 'W', 'B']

    localization = MonteCarloLocalization(world=colours, measurements=measurements, movements=movements, sensor_prob_correct=1.0)
    posterior = localization.compute_posterior()
