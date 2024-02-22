# Linear Regression Forward Implementation

## Overview
This Python code implements linear regression, a fundamental technique in machine learning used for predicting continuous values. Linear regression aims to establish a linear relationship between independent variables (features) and a dependent variable (target).

## Libraries Used
- **NumPy**: NumPy is a powerful library for numerical computing in Python. It provides support for large, multi-dimensional arrays and matrices, along with a collection of mathematical functions to operate on these arrays efficiently.

## Functions Implemented

### get_model_prediction(X, weights)
- **Description**: This function computes the predicted values for each data point in the dataset using the linear regression model.
- **Inputs**:
  - `X`: The dataset containing the independent variables. It is represented as an **_Nx3_** NumPy array, where N is the number of data points and each row represents a data point with three features.
  - `weights`: The current weights (coefficients) of the linear regression model. It is represented as a **_3x1_** NumPy array containing the weights for each feature.
- **Output**: 
  - An **_Nx1_** NumPy array containing the predicted values for each data point.

### get_error(model_prediction, ground_truth)
- **Description**: This function calculates the mean squared error between the predicted values obtained from the linear regression model and the actual ground truth values.
- **Inputs**:
  - `model_prediction`: The predicted values obtained from the linear regression model for each data point. It is represented as an **_Nx1_** NumPy array.
  - `ground_truth`: The true target values for each data point. It is represented as an **_Nx1_** NumPy array.
- **Output**:
  - The mean squared error between the predicted values and the ground truth values.

## Formulae
### Linear Regression Model Prediction
The predicted value for each data point is computed using the formula:<br>
&nbsp;&nbsp;&nbsp;&nbsp;**_predicted_value = w1 * x1 + w2 * x2 + w3 * x3_**<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Where:<br>
    - `w1`, `w2`, and `w3` are the weights (coefficients) of the linear regression model.
    - `x1`, `x2`, and `x3` are the features of the data point.

### Mean Squared Error (MSE)
The mean squared error (MSE) is calculated as the average of the squared differences between the predicted values and the ground truth values:<br>
&nbsp;&nbsp;&nbsp;&nbsp;**_MSE = (1 / N) * Σ(predicted_value - ground_truth)^2_**<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Where:<br>
    - `N` is the number of data points.
    - `Σ` represents the summation over all data points.
