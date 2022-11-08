import torch
import torch.nn.functional as F
import torch.nn as nn
import pandas as pd
import csv
import os
import numpy
from sklearn.model_selection import train_test_split
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
learning_rate = 0.01

PATH = r"C:\Users\Tatonta\Desktop\Matlab"

MegaVector = pd.read_csv(os.path.join(PATH,"xyzVectors.csv")).to_numpy(dtype="float64")
targets = pd.read_csv(os.path.join(PATH,"targets.csv")).to_numpy(dtype="float64")
n_features = MegaVector.shape[0]

MegaVector_t = torch.from_numpy(MegaVector).to(device)
targets_t = torch.from_numpy(targets).to(device)


class ParametersPredictor(nn.Module):
    def __init__(self, n_input_features):
        super(ParametersPredictor, self).__init__()
        self.sequential = nn.Sequential(nn.Linear(n_input_features, 1000),
        nn.Linear(1000, 1000),
        nn.Linear(1000, 1000),
        nn.Linear(1000, 1000),
        nn.Linear(1000, 3)
        )


    def forward(self, x):
        out = self.sequential(x)
        return out


NeuralNetwork = ParametersPredictor(n_features).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(NeuralNetwork.parameters(), lr=learning_rate)
EPOCHS = 100
for epoch in range(EPOCHS):
    for i in tqdm(range(302)):  # from 0, to the len of x, stepping BATCH_SIZE at a time. [:50] ..for now just to dev
        # print(f"{i}:{i+BATCH_SIZE}")
        predicted_y = NeuralNetwork.forward(MegaVector_t[:,i].float())
        loss = criterion(targets_t[i].float(), predicted_y.float())

        # backward pass
        optimizer.zero_grad()
        loss.backward()

        # updates
        optimizer.step()

        # zero gradients
    print(f"Epoch: {epoch}. Loss: {loss:.8f}")

print(NeuralNetwork.forward(MegaVector_t[:,1]))