from localization import Localization
from movement import Movement

if __name__ == '__main__':
    colours = ['B', 'W', 'B', 'B', 'B']

    start_position = 0
    movements = [Movement.RIGHT, Movement.LEFT]
    measurements = ['W', 'B']

    localization = Localization(world=colours, measurements=measurements, movements=movements, sensor_prob_correct=1.0)
    posterior = localization.calculate_probabilities()
