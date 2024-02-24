import torch

# Create a tensor
tensor1 = torch.tensor([[1, 2], [3, 4]])
print(tensor1)

tensor2 = torch.tensor([[5, 6], [7, 8]])

# Element-wise addition
added_tensor = tensor1 + tensor2
print(added_tensor)

# Matrix multiplication
matmul_result = torch.matmul(tensor1, tensor2)
print(matmul_result)

# Create a tensor of zeros
zeros_tensor = torch.zeros(2, 3)
print(zeros_tensor)

# Create a tensor of ones
ones_tensor = torch.ones(2, 3)
print(ones_tensor)

# Compute column-wise sum
col_sum = torch.sum(tensor1, axis=0)
print(col_sum)

# Compute row-wise sum
row_sum = torch.sum(tensor1, axis=1)
print(row_sum)

# Create a tensor of ones with shape (3, 1)
third_tensor = torch.ones(3, 1)
print(third_tensor.shape)

# Squeeze the tensor to remove dimensions of size 1
squeezed_tensor = torch.squeeze(third_tensor)
print(squeezed_tensor.shape)

# Unsqueeze the tensor to add a dimension of size 1
unsqueezed_tensor = torch.unsqueeze(squeezed_tensor, dim=1)
print(unsqueezed_tensor.shape)
