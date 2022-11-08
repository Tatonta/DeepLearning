import torch
import numpy as np

X = np.array([1,2,3,4,5,6,7,8,9,10], dtype=np.float32)
y = np.array([2,4,6,8,10,12,14,16,18,20], dtype=np.float32)
w = 0.00

def forward(train_data):
    return w*train_data


def loss(y,y_predicted):
    return ((y_predicted-y)**2).mean()
# print((X*w-y)**2)
# Chain rule derivative of: d/dx[1/N*(x*w-y)^2]=1/N*2x(x*w-y)
def gradient(x,y,y_pred):
    return (2*x*(y_pred-y)).mean()

num_epochs = 10
learning_rate = 0.01
print(f"Prediction before training: f(5) = {forward(5):.3f}")
# Training
for epoch in range(num_epochs):
    # prediction
    y_pred = forward(X)

    # loss
    l = loss(y,y_pred)

    dw = gradient(X,y,y_pred)

    #update weights
    w -= learning_rate * dw


    if epoch % 1 == 0:
        print(f"epoch {epoch+1}: w = {w:.3f}, loss= {l:.8f}")

print(f"Prediction after training: f(5) = {forward(5):.3f}")