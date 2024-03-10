from movement import Movement


class Localization:
    def __init__(self, world: [str], measurements, movements, prob_hit, prob_miss):
        self.world = world
        self.measurements = measurements
        self.movements = movements
        self.prob_hit = prob_hit
        self.prob_miss = prob_miss
        # initialize beliefs with equal probabilities
        self.probabilities = [1.0 / float(len(world))] * len(world)
        self.show(self.probabilities)

    def move(self, old_belief, movement: [Movement]):
        new_belief = [0.0 for _ in range(len(self.world))]

        for i in range(len(old_belief)):
            movement_value = movement.value

            new_i = (i - movement_value.x_delta) % len(old_belief)

            new_belief[i] = old_belief[new_i]

        return new_belief

    def sense(self, old_belief, world, measurement):
        new_belief = [0.0 for _ in range(len(self.world))]
        sigma = 0.0

        for i in range(len(old_belief)):
            hit = (measurement == world[i])

            if hit:
                new_belief[i] = old_belief[i] * self.prob_hit
            else:
                new_belief[i] = old_belief[i] * self.prob_miss

            sigma += new_belief[i]

        for i in range(len(new_belief)):
            new_belief[i] /= sigma

        return new_belief

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
