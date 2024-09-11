import numpy as np
import pandas as pd

class Horner():
    
    def __init__(self, coef):
        self.coef = np.array(coef)
        self.n = len(coef)
    
    def horner_polyomial_value_at_point(self, point):
        result = 0
        for coef in self.coef:
            result = result * point + coef
        return result
    
    def horner_polynomial_divider(self, point):
        n = len(self.coef)
        result = np.zeros(n)
        result[0] = self.coef[0]
        for i in range(1, n):
            result[i] = result[i - 1] * point + self.coef[i]
        return result
    
    def horner_multiplication(self, point):
        result = [self.coef[0]]
        for i in range(1, self.n):
            result.append(self.coef[i] - self.coef[i - 1] * point)
        result.append(-self.coef[-1] * point)
        return result
    
    def derivative(self):
        result = np.zeros(self.n - 1)
        for i in range(self.n - 1):
            result[i] = self.coef[i] * (self.n - i - 1)
        return result
    