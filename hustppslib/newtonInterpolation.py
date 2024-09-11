import sys
sys.path.append("D:\\Specialization\\Mathematics Specialization\\Phương pháp số\\hustppslib\\horner")

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from horner import Horner

class NewtonwInterpolation():

    def __init__(self, x, y):
        self.x = np.array(x)
        self.y = np.array(y)
        self.n = len(x)
        self.coef = np.zeros((self.n, self.n))

    def divided_difference_forward(self, x, y):
        n = len(x)
        coef = np.zeros((n, n))
        coef[:, 0] = y # First column is y

        # Calculate the matrix of divided differences
        for j in range(1, n):
            for i in range(n - j):
                coef[i, j] = (coef[i + 1, j - 1] - coef[i, j - 1]) / (x[i + j] - x[i])
        return coef

    def divided_difference_backward(self, x, y):
        n = len(x)
        coef = np.zeros((n, n))
        coef[:, 0] = y # First column is y

        # Calculate the matrix of divided differences
        for j in range(1, n):
            for i in range(n - j):
                coef[i, j] = (coef[i, j - 1] - coef[i - 1, j - 1]) / (x[i] - x[i - j])
        return coef
    
    def padding_matrix(self, coef, length):
        n = self.coef.shape[0]
        pad = np.zeros((length - n, n))
        return np.vstack((coef, pad))
    
    def newton_interpolation_forward(self, x, y, type='forward'):
        
        # Compute the devided differences for Newton's forward interpolation
        coefficient_for_polynomial = np.array([self.divided_difference_forward(self.x, self.y)[0, i] for i in range(self.n)])
        if type == 'backward':
            coefficient_for_polynomial = np.array([self.divided_difference_backward(self.x, self.y)[0, i] for i in range(self.n)])

        # Initialize an array to store interpolation terms in descending order
        coefficients = np.zeros(self.n)

        # Initialize the current interpolation term
        current_term = np.array([1])
        horner = Horner(current_term)
        # Initialize a matrix to store the coefficient of each term for each degree
        terms_matrix = [[self.x[0]]]

        # Calculate interpolation terms
        for index, point in enumerate(self.x[:-1]):
            current_term = np.array(horner.horner_multiplication(point)) # Use Horner schema to calculate polynomial multiplication
            terms_matrix.append(coefficient_for_polynomial[index + 1] * current_term)

        # Pad the matrix to ensure each row has the same length
        terms_matrix = self.padding_matrix(terms_matrix, self.n)

        # Each column sum represents the coefficient of the corresponding degree
        coefficients = terms_matrix.sum(axis=0)

        return coefficients
    
    def plot_interpolation(self):
        x = np.linspace(self.x[0], self.x[-1], 100)
        y = self.newton_interpolation(x)
        plt.plot(x, y, label='Interpolation')
        plt.scatter(self.x, self.y, color='red', label='Data points')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.show()

newtonSolver = NewtonwInterpolation([2.1, 2.2, 2.3, 2.4, 2.5, 2.6], [3.071161, 3.1096173, 3.138985, 3.1599698, 3.1733613, 3.1800247])
print(newtonSolver.newton_interpolation_forward(newtonSolver.x, newtonSolver.y, 'forward'))