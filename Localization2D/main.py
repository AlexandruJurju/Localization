from localization import Localization
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

    localization = Localization(world=colours, measurements=measurements, movements=movements, robot_position=start_position, prob_hit=1.0, prob_miss=0.0)
    posterior = localization.compute_posterior()
