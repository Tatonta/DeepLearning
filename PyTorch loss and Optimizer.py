# 1) Design model (input, output size, forward pass)
# 2) Construct loss and optimizer
# 3) Training loop
# - Forward pass : compute prediction
# - backward pass : gradients
# - update weights

import torch
import torch.nn as nn
import numpy as np

X = torch.tensor([[1],[2],[3],[4],[5],[6],[7],[8],[9],[10]], dtype=torch.float32)
y = torch.tensor([[2],[4],[6],[8],[10],[12],[14],[16],[18],[20]], dtype=torch.float32)

X_test = torch.tensor([5], dtype=torch.float32)

n_samples, n_features = X.shape

input_size = n_features
output_size = n_features

#model = nn.Linear(input_size, output_size)

class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()
        #define layers
        self.lin = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.lin(x)

model = LinearRegression(input_size, output_size)

# model prediction
# def forward(train_data):
#     return w*train_data


# def loss(y,y_predicted):
#     return ((y_predicted-y)**2).mean()
# print((X*w-y)**2)
# Chain rule derivative of: d/dx[1/N*(x*w-y)^2]=1/N*2x(x*w-y)

num_epochs = 10000
learning_rate = 0.01

loss = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
print(f"Prediction before training: f(5) = {model(X_test).item():.3f}")
# Training
for epoch in range(num_epochs):
    # prediction
    y_pred = model(X)

    # loss
    l = loss(y,y_pred)

    l.backward() # gradient of the loss

    #update weights
    # with torch.no_grad():
    #     w -= learning_rate * w.grad()
    optimizer.step()

    # zero gradient
    optimizer.zero_grad()

    if epoch % 1 == 0:
        [w, b] = model.parameters()
        print(f"epoch {epoch+1}: w = {w[0][0].item():.3f}, loss= {l:.8f}")

print(f"Prediction after training: f(5) = {model(X_test).item():.3f}")