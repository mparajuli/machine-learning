'''
Problem: 

Your task is to minimize the function via Gradient Descent: f(x) = x^2

Given the number of iterations to perform gradient descent, the learning rate, and an initial guess,
return the value of x that globally minimizes this function.

Round your final result to 5 decimal places using Python's round() function.
'''

class Gradient_Descent:
    @staticmethod
    def get_minimizer(iterations, learning_rate, init): 
        """
        Performs gradient descent to minimize the function f(x) = x^2.

        Args:
            iterations (int): The number of iterations for gradient descent.
            learning_rate (float): The learning rate for gradient descent.
            init (float): The initial guess for x.

        Returns:
            float: The value of x that globally minimizes the function.
        """
        
        for i in range(iterations):
            gradient = 2 * init
            init = init - learning_rate * gradient

        return round(init, 5)  

# Test cases
# Case 1: iterations = 0, learning_rate = 0.1, init = 10
# Expected output: 10 (unchanged)
print(Gradient_Descent.get_minimizer(0, 0.1, 10))

# Case 2: iterations = 5, learning_rate = 0.1, init = 10
# Expected output: 3.2768 (closer to 0, as gradient descent approaches the minimum of f(x) = x^2)
print(Gradient_Descent.get_minimizer(5, 0.1, 10))
