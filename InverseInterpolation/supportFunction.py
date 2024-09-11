import numpy as np

def differences_forward(interpolation_values):
    """
    Function to compute differences for Newton forward interpolation

    Args:
    interpolation_points (np.array):
    interpolation_values (np.array):

    Returns:
    (np.array): differences forward table for Newton interpolation
    """
    # Size of matrix
    n = len(interpolation_values)

    # Create divided differences matrix
    differences = np.zeros([n, n])

    # the first column is y
    differences[:, 0] = interpolation_values

    # Calculate the matrix
    for j in range(1, n):
        for i in range(n - j):
            differences[i, j] = (differences[i + 1, j - 1] - differences[i, j - 1])

    return differences

def horner_multiplication(coefficients, value):
    """
    Compute the result of multiplying the original polynomial by (x - value) using Horner's method.

    Args:
    coefficients (array): An array of coefficients in descending order of degree.
    value (float): The value by which to multiply the polynomial.

    Returns:
    array: An array of coefficients representing the result of multiplying the polynomial.
    """
    mul_coefficients = [coefficients[0]]
    for i in range(1, len(coefficients)):
        mul_coefficients.append(coefficients[i] - coefficients[i - 1] * value)
    mul_coefficients.append(-coefficients[-1] * value)
    return mul_coefficients

def padding_matrix(matrix, length):
    """
    To padding the array of an matrix to an fixed length, make calculation
    of column sum easier

    Args:
    matrix (np.array): matrix to padding
    length (int): fixed length to pad
    """
    for index, array in enumerate(matrix):
        if len(array) < length:
            padding = np.zeros(length - len(array))
            matrix[index] = np.concatenate((padding, array))
    return np.array(matrix)