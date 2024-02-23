# Linear Regression Training

## Problem Description
Given `get_model_prediction()` and `get_derivative()` functions for a model, implement the training loop to perform gradient descent. The goal is to find the optimal weights for the model that minimize the error between the predicted values and the actual values.

## Input
- `X`: The dataset for training the model. `X.length = n` and `X[i].length = 3` for `0 <= i < n`.
- `Y`: The correct answers from the dataset. `Y.length = n`.
- `num_iterations`: The number of iterations to run gradient descent for. `num_iterations > 0`.
- `initial_weights`: The initial weights for the model (`w1`, `w2`, `w3`). `initial_weights.length = 3`.

## Output
Return the final weights after training in the form of a NumPy array with dimension 3.

## Implementation Details
- `get_derivative(model_prediction, ground_truth, N, X, desired_weight)`: Computes the derivative of the loss function with respect to a specific weight of the model. This derivative is crucial for updating the weights during the training process.
- `get_model_prediction(X, weights)`: Computes the model predictions for the given dataset `X` and weights. This function provides the predicted values of the target variable based on the current model weights.
- `learning_rate`: The learning rate used for gradient descent. It's a hyperparameter that controls the size of the steps taken during the optimization process. In this implementation, it's set to `0.01`.
- `train_model(X, Y, num_iterations, initial_weights)`: Implements the training loop using gradient descent. It iteratively updates the weights of the model by computing the derivatives and applying the update rule.

### Update Formula
The update rule for gradient descent used in the training loop is as follows:<br>
&nbsp;&nbsp;&nbsp;&nbsp;_For each weight `i`:_<br>
&nbsp;&nbsp;&nbsp;&nbsp;***_initial_weights[i] -= learning_rate * derivative_i_***
This rule adjusts each weight in the direction that minimizes the loss function, with the step size determined by the learning rate.
