'''
Problem: Your task is to implement linear regression.

Implement get_model_prediction() which returns a prediction value for each dataset value, 
and get_error() which calculates the error for given prediction data.

Inputs - get_model_prediction:
X - the dataset to be used by the model to predict the output. len(X) = n, and len(X[i]) = 3 for 0 <= i < n.
weights - the current w1, w2, and w3 weights for the model. len(weights) = 3.

Inputs - get_error:
model_prediction - the model's prediction for each training example. len(model_prediction) = n.
ground_truth - the correct answer for each example. len(ground_truth) = n.
'''

import numpy as np

class LinearRegForward:
  def get_model_prediction(self, X, weights):  
    # X is an Nx3 NumPy array
    # weights is a 3x1 NumPy array
    model_prediction = np.matmul(X, weights)
    return np.round(model_prediction, 5)
  
  def get_error(self, model_prediction, ground_truth): 
    # model_prediction is an Nx1 NumPy array
    # ground_truth is an Nx1 NumPy array
    diff = model_prediction - ground_truth
    squared = np.square(diff)
    error = np.mean(squared)
    return np.round(error, 5)


'''
For test purpose only
'''

X_test = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])  
weights_test = np.array([0.1, 0.2, 0.3]) 
ground_truth_test = np.array([14, 32, 50]) 

# Instantiate LinearRegForward
linear_reg = LinearRegForward()

# Test get_model_prediction()
model_predictions = linear_reg.get_model_prediction(X_test, weights_test)
expected_predictions = np.array([1*0.1 + 2*0.2 + 3*0.3, 4*0.1 + 5*0.2 + 6*0.3, 7*0.1 + 8*0.2 + 9*0.3])
assert np.allclose(model_predictions, expected_predictions), "Model predictions do not match expected values"

# Test get_error()
error = linear_reg.get_error(model_predictions, ground_truth_test)
expected_error = np.mean(np.square(model_predictions - ground_truth_test))
assert np.isclose(error, expected_error), "Error calculation is incorrect"

print("All tests passed successfully!")
