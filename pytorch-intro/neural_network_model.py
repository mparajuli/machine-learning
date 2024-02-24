import torch.nn as nn

class NeuralNetworkModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Define layers
        self.input_layer = nn.Linear(3, 4)   # Input layer with 3 input features and 4 output features
        self.hidden_layer = nn.Linear(4, 4)  # Hidden layer with 4 input features and 4 output features
        self.output_layer = nn.Linear(4, 2)  # Output layer with 4 input features and 2 output features

    def forward(self, x):
        # Define forward pass
        return self.output_layer(self.hidden_layer(self.input_layer(x)))

# Instantiate the model
model = NeuralNetworkModel()

# Now, we can train the model

# After training, we can use the model to get predictions
