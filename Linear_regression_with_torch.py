import torch
import numpy as np

X = torch.tensor([1,2,3,4,5,6,7,8,9,10], dtype=torch.float32)
y = torch.array([2,4,6,8,10,12,14,16,18,20], dtype=torch.float32)
w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

def forward(train_data):
    return w*train_data


def loss(y,y_predicted):
    return ((y_predicted-y)**2).mean()
# print((X*w-y)**2)
# Chain rule derivative of: d/dx[1/N*(x*w-y)^2]=1/N*2x(x*w-y)

num_epochs = 10
learning_rate = 0.01
print(f"Prediction before training: f(5) = {forward(5):.3f}")
# Training
for epoch in range(num_epochs):
    # prediction
    y_pred = forward(X)

    # loss
    l = loss(y,y_pred)

    l.backward() # gradient of the loss

    #update weights
    with torch.no_grad():
        w -= learning_rate * w.grad()

    w.grad.zero_()

    if epoch % 1 == 0:
        print(f"epoch {epoch+1}: w = {w:.3f}, loss= {l:.8f}")

print(f"Prediction after training: f(5) = {forward(5):.3f}")