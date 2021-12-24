import math

class MathFunctions:
    def __init__(self):
        pass

    @classmethod
    def ReLU(self, x : float) -> float:
        return max(0, x)

    @classmethod
    def leakyReLU(self, x : float) -> float:
        return max(0.1 * x, x)

    @classmethod
    def sigmoid(self, x : float, diff=False) -> float:
        if diff:
            return self.sigmoid(x) * (1.0 - self.sigmoid(x))
        else:
            return 1.0 / (1.0 + math.exp(-x))
