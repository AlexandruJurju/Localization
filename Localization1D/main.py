from localization import Localization
from movement import Movement

if __name__ == '__main__':
    colours = ['B', 'W', 'B', 'B', 'B']

    start_position = 0
    movements = [Movement.RIGHT, Movement.LEFT]
    measurements = ['W', 'B']

    localization = Localization(world=colours, measurements=measurements, movements=movements, robot_position=start_position, prob_hit=0.8, prob_miss=0.2)
    posterior = localization.calculate_probabilities()
