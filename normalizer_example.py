# Welford's Algorithm
import numpy as np
class ObservationNormalizer:
    def __init__(self, epsilon=1e-8):
        self.mean = 0
        self.var = 0
        self.count = 0
        self.epsilon = epsilon

    def update(self, obs):
        self.count += 1
        delta = obs - self.mean
        self.mean += delta / self.count
        delta2 = obs - self.mean
        self.var += delta * delta2

    def normalize(self, obs):
        return (obs - self.mean) / (np.sqrt(self.var / self.count) + self.epsilon)


# Использование
normalizer = ObservationNormalizer()
obs = np.array([1.0, 2.0, 3.0])

# Обновляем нормализатор для каждого элемента отдельно
for o in obs:
    normalizer.update(o)

norm_obs = normalizer.normalize(obs)
print(norm_obs)
