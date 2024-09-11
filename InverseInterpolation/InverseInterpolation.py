import numpy as np
import matplotlib.pyplot as plt
from supportFunction import differences_forward, horner_multiplication, padding_matrix
import math 
from scipy.interpolate import lagrange

class InverseInterpolation():
    """
    This class is used to perform inverse interpolation on a set of points.

    """
    def __init__(self, interpolation_points, interpolation_values, y_value):
        self.interpolation_points = interpolation_points
        self.interpolation_values = interpolation_values
        self.y_value = y_value

    def root_interval(self):
        """
        Find all root intervals in the given interpolation values range
        (root interval is an interval in which the values of interpolation
        change their sign)

        Args:
        -- self: the object of the class

        Returns: root intervals
        """
        # Check if the input array is not empty
        if len(self.interpolation_values) == 0:
            return []
        
        # Initialize an empty list to store root intervals
        root_intervals = []
        current_interval = []

        # Iterate through the interpolation values
        for idx, value in enumerate(self.interpolation_values):
            if idx == len(self.interpolation_values) - 1:
                break
            # Interval contains 2 consecutive values
            current_interval = [self.interpolation_values[idx], self.interpolation_values[idx + 1]]
            # print(interpolation_values[idx], interpolation_values[idx + 1])
            # Check if the y_value is within the current interval
            if np.min(current_interval) <= self.y_value <= np.max(current_interval):
                # Add the root interval to the list
                root_intervals.append([idx, idx + 1])

        return root_intervals

    def get_extendable(self):
        """
        Check if the root intervals can be extended to make an interpolation 
        range long enough.
        The condition to add to the root interval is that it is monotonous
        in the interval after adding.
        """
        n = len(self.interpolation_values)
        root_intervals = self.root_interval()
        inv_intervals = [] # Interval can apply inverse function method
        eqldis_intervals = [] # Interval will apply iterative method

        # Check if the root intervals is in the middle so it can extended and otherwise
        # it can't
        for index, interval in enumerate(root_intervals):
            first_element = self.interpolation_values[interval[0]]
            last_element = self.interpolation_values[interval[-1] + 1]
            check_first = np.where(self.interpolation_values == first_element)[0][0]
            check_last = np.where(self.interpolation_values == last_element)[0][0]
            if check_first < 2 or check_last > n - 2:
                eqldis_intervals.append(interval)
            elif check_first >= 2 and check_last <= n + 2:
                inv_intervals.append(interval)

        return inv_intervals, eqldis_intervals
    
    def extend_interval(self, extendable_intervals, min_interval_length=5):
        """
        Expand the root interval equally to both sides to make an interpolation range long enough.
        """
        # Initialize of extended intervals
        extended_intervals = []

        # Iterate through the root intervals
        for index, interval in enumerate(extendable_intervals):
            # Left and right index
            left_idx = interval[0]
            right_idx = interval[-1]

            # Iterate through the interval
            while len(interval) < min_interval_length and (left_idx != 0 and right_idx != len(self.interpolation_values)):

                # Check left and right if both satisfy the condition to add
                if self.interpolation_values[right_idx] > self.interpolation_values[left_idx]:
                    check_left = self.interpolation_values[left_idx - 1] < self.interpolation_values[left_idx]
                    check_right = self.interpolation_values[right_idx + 1] > self.interpolation_values[right_idx]
                elif self.interpolation_values[right_idx] < self.interpolation_values[left_idx]:
                    check_left = self.interpolation_values[left_idx - 1] > self.interpolation_values[left_idx]
                    check_right = self.interpolation_values[right_idx + 1] < self.interpolation_values[right_idx]

                # Extend the interval
                if check_left and check_right:
                    # Add to right
                    interval.append(right_idx + 1)
                    right_idx += 1
                    # Add to left
                    interval.insert(0, left_idx - 1)
                    left_idx -= 1
                else:
                    break
            # Update
            extended_intervals.append(interval)

        return extended_intervals

    def get_monotonuos_interval(self, min_interval_length=5):
        """
        Expand the root interval equally to both sides to make an interpolation range long enough.
        The condition to add to the root interval is that it is monotonous
        in the interval after adding. Stop the addition if the length of the extended interval
        is greater than min_interval_length.
        
        Returns:
        np.array: set of interval will apply inverse method and set of interval will apply iterative method
        """

        # Get the root intervals
        inv_intervals, eqldis_intervals = self.get_extendable()
        extended_intervals = self.extend_interval(inv_intervals, min_interval_length)
        temp = extended_intervals.copy()

        # If not satisfy the condition to apply inverse function method, add to
        # equal distance set
        for index, interval in enumerate(extended_intervals):
            if len(interval) < min_interval_length:
                eqldis_intervals.append(interval)
                temp.remove(interval)

        extended_intervals = temp
        return extended_intervals, eqldis_intervals
    
    def phi_t(self, interval):
        """
        Function of t to use iterative method to find t_bar

        Return:
        (np.array): Coefficients of phi_t in descending order
        """
        n = len(interval)  # Determine the degree of the polynomial

        # Compute the divided differences for Newton's forward interpolation
        coefficient_for_polynomial = np.array([differences_forward(interval)[0, i] for i in range(n)])

        # Initialize an array to store interpolation terms in descending order
        coefficients = np.zeros(n)

        # Initialize the current interpolation term
        current_term = np.array([1])

        # Initialize a matrix to store the coefficient of each term for each degree
        first_difference = coefficient_for_polynomial[1]
        terms_matrix = [[(self.y_value - interval[0]) / first_difference]]

        # Calculate interpolation terms
        for index, point in enumerate(interval[:-1]):
            if index == 0:
                current_term = np.array([1, 0])
                terms_matrix.append([0] * n)
            else:
                current_term = np.array(horner_multiplication(current_term, index)) # Use Horner schema to calculate polynomial multiplication
                terms_matrix.append(np.power(-1, index) * (coefficient_for_polynomial[index + 1] / (math.factorial(index + 1) * first_difference)) * current_term)

        # Pad the matrix to ensure each row has the same length
        terms_matrix = padding_matrix(terms_matrix, n)

        # Each column sum represents the coefficient of the corresponding degree
        coefficients = terms_matrix.sum(axis=0)

        return coefficients
    
    def shift_iterative_intervals(self, eqldis_intervals):

        for i, interval in enumerate(eqldis_intervals):
            if len(interval) == 4:
                eqldis_intervals[i] = [interval[1],interval[-1] + 4]
        return eqldis_intervals
        
    def iterative_method(self, interval, tol=1e-7, max_iterations=100):
        """
        Iterative method to find the root of the inverse interpolation

        Args:
        - interpolation_values (np.array): Array of y values corresponding to interpolation points
        - y"_value (float): The target y value for inverse interpolation
        - tol (float): Tolerance for stopping criterion (default: 1e-2)
        - max_iterations (int): Maximum number of iterations (default: 100)

        Returns:
        - float: the value to add to find root of the inverse interpolation

        Raises:
        - RuntimeError: If the iterative method does not converge within the maximum number of iterations
        """
        # Initial approximation
        first_difference = interval[1] - interval[0]
        t = (self.y_value - interval[0]) / first_difference

        # phi(t) function
        phi_t_f = self.phi_t(interval)

        for _ in range(max_iterations):
            t_new = np.polyval(phi_t_f, t)
            if abs(t_new - t) < tol:
                return t_new  # The solution is close enough
            t = t_new
        return t

    def inverse_interpolation(self):
        """
        Solve the inverse interpolation problem using Newton interpolation
        polynomial for both scenerio

        Args:
        - interpolation_points (np.array): Interpolation_points
        - interpolation_values (np.array): Corresponding value to the interpolation
        points
        - y_value (float): values to find correspond xbar

        Returns:
        (dictionary): key are y_interval, value are root
        """
        # Get the intervals can apply inverse function method
        # And the intervals can apply iterative method
        inverse_intervals, equidistant_intervals = self.get_monotonuos_interval()
        # Initialize set contains root of the polynomail at y_value
        roots = {}

        # Loop through the set can apply inverse function method
        for interval in inverse_intervals:
            intplt_values = self.interpolation_values[interval[0]: interval[-1] + 1]
            interval_of_x = self.interpolation_points[interval[0]: interval[-1] + 1]
            coefficients = lagrange(intplt_values,interval_of_x)
            roots[f"[{interval_of_x}]"] = (np.polyval(coefficients, self.y_value))

        # Loop through the set can apply iterative method
        if len(equidistant_intervals) ==  0:
            return roots 
        equidistant_intervals = self.shift_iterative_intervals(equidistant_intervals)

        for interval in equidistant_intervals:
            points = self.interpolation_points[interval[0]: interval[-1] + 1]
            values = self.interpolation_values[interval[0]: interval[-1] + 1]
            tbar = self.iterative_method(values, tol=1e-4, max_iterations=1000)
            distance = np.diff(points)
            xbar = points[0] + distance[0] * tbar
            roots[f"[ {points}]"] = (xbar)

        return roots