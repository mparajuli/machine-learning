'''
Problem: Given get_model_prediction() and get_derivative() functions for a model, 
implement the training loop to perform gradient descent.

Your goal is to implement the train_model() function, which has the following as input:

X: The dataset for training the model. X.length = n and X[i].length = 3 for 0 <= i < n.
Y: The correct answers from the dataset. Y.length = n.
num_iterations: The number of iterations to run gradient descent for. num_iterations > 0.
initial_weights: The initial weights for the model (w1, w2, w3). initial_weights.length = 3.

Return the final weights after training in the form of a NumPy array with dimension 3.
'''

import numpy as np

class LinearRegressionTraining:
  # get_derivatice() is given to you
  def get_derivative(self, model_prediction, ground_truth, N, X, desired_weight):
        # note that N is just len(X)
        return -2 * np.dot(ground_truth - model_prediction, X[:, desired_weight]) / N

  # get_model_prediction() is given to you
  def get_model_prediction(self, X, weights):
        return np.squeeze(np.matmul(X, weights))
  
  # learning_rate is given to you
  learning_rate = 0.01

  def train_model(self, X, Y, num_iterations, initial_weights):
       
    for i in range(num_iterations):
        model_prediction = self.get_model_prediction(X, initial_weights)

        # Get the derivative for each weight
        d1 = self.get_derivative(model_prediction, Y, len(X), X, 0)
        d2 = self.get_derivative(model_prediction, Y, len(X), X, 1)
        d3 = self.get_derivative(model_prediction, Y, len(X), X, 2)

        # Using gradient descent update rule
        initial_weights[0] -= self.learning_rate * d1
        initial_weights[1] -= self.learning_rate * d2
        initial_weights[2] -= self.learning_rate * d3
    
    return np.round(initial_weights, 5)

'''
Test Case:
Input:
X = [[1, 1, 1], [2, 3, 4]]
Y = [3, 9]
num_iterations = 10
initial_weights = [0.3, 0.2, 0.8]

Output:
[0.64135 0.67769 1.41403]
'''

X = np.array([[1, 1, 1], [2, 3, 4]])
Y = [3, 9]
num_iterations = 10
initial_weights = [0.3, 0.2, 0.8]

# Create an instance of the LinearRegressionTraining class
trainer = LinearRegressionTraining()

# Call the train_model function on the instance
print(trainer.train_model(X, Y, num_iterations, initial_weights))
