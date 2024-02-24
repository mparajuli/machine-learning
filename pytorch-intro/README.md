# PyTorch 

PyTorch is a popular open-source machine learning library developed by Facebook's AI Research lab. It's primarily used for building deep learning models, including neural networks, and it provides tools for tensor computation with strong GPU acceleration support. Here are some fundamental concepts and functions in PyTorch.

## PyTorch Basics

### Creating Tensors

PyTorch tensors are multi-dimensional arrays similar to NumPy arrays but with GPU acceleration capabilities.

- **Function:** `torch.tensor()`
- **Syntax:** `torch.tensor(data, dtype=None, device=None, requires_grad=False)`
- **Parameters:**
  - `data`: Input data, can be a list, tuple, NumPy array, etc.
  - `dtype` (optional): Data type of the tensor (e.g., `torch.float`, `torch.int`)
  - `device` (optional): Specifies the device (e.g., CPU or GPU) where the tensor will be allocated.
  - `requires_grad` (optional): If `True`, tracks operations on the tensor for computing gradients.

### Element-wise Addition

- **Operation:** Element-wise addition of two tensors.
- **Syntax:** `result_tensor = tensor1 + tensor2`

### Matrix Multiplication

- **Operation:** Computes the matrix product of two tensors.
- **Syntax:** `result_tensor = torch.matmul(tensor1, tensor2)`

### Creating Zeros and Ones Tensors

- **Functions:** `torch.zeros()`, `torch.ones()`
- **Syntax:**
  - `zeros_tensor = torch.zeros(shape)`
  - `ones_tensor = torch.ones(shape)`
- **Parameter:**
  - `shape`: Shape of the tensor (e.g., `(2, 3)` for a 2x3 tensor)

### Computing Sums

- **Function:** `torch.sum()`
- **Syntax:** `sum_tensor = torch.sum(input_tensor, dim=None)`
- **Parameters:**
  - `input_tensor`: Input tensor.
  - `dim` (optional): Dimension along which to compute the sum.

### Modifying Tensor Shape

- **Functions:** `torch.squeeze()`, `torch.unsqueeze()`
- **Syntax:**
  - `squeezed_tensor = torch.squeeze(input_tensor)`
  - `unsqueezed_tensor = torch.unsqueeze(input_tensor, dim)`

## Neural Network Model

A simple neural network model is defined using PyTorch's neural network module (`torch.nn`).

- **Module:** `torch.nn.Module`
- **Syntax:** `class NeuralNetworkModel(nn.Module):`
- **Layers:** Linear layers defined using `nn.Linear()`
- **Forward Pass:** Defined in the `forward()` method of the model class.
