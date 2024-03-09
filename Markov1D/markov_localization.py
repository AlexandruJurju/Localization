from Movement import Movement


class MonteCarloLocalization:
    def __init__(self, world: [str], measurements, movements, sensor_prob_correct):
        self.world = world
        self.measurements = measurements
        self.movements = movements
        self.sensor_prob_correct = sensor_prob_correct
        self.probabilities = [1.0 / float(len(world))] * len(world)
        self.show(self.probabilities)

    def move(self, prior, movement: [Movement]):
        posterior = [0.0 for _ in range(len(self.world))]

        for i in range(len(prior)):
            movement_value = movement.value

            new_i = (i - movement_value.x_delta) % len(prior)

            posterior[i] = prior[new_i]

        return posterior

    def sense(self, prior, world, measurement):
        posterior = [0.0 for _ in range(len(self.world))]
        total_probability = 0.0

        for i in range(len(prior)):
            hit = (measurement == world[i])

            if hit:
                posterior[i] = prior[i] * self.sensor_prob_correct
            else:
                posterior[i] = prior[i] * (1 - self.sensor_prob_correct)

            total_probability += posterior[i]

        for i in range(len(posterior)):
            posterior[i] /= total_probability

        return posterior

    def calculate_probabilities(self):
        p = self.probabilities
        for i in range(len(self.measurements)):
            p = self.move(p, self.movements[i])

            p = self.sense(p, self.world, self.measurements[i])
            self.show(p)

        return p

    @staticmethod
    def show(p):
        """Prints the probability distribution"""
        formatted_probs = '[' + ','.join(map(lambda x: '{0:.5f}'.format(x), p)) + ']'
        print(formatted_probs + "\n")
