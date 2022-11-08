import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as T
from torchvision import datasets
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import make_grid
from torchvision.utils import save_image
from IPython.display import Image
import matplotlib.pyplot as plt
import numpy as np
import os

def translating_data(numero):
    dizionario={0:"Alilaguna", 1:"Ambulanza", 2:"Barchino", 3:"Cacciapesca", 4:"Caorlina", 5:"Gondola", 6:"Lanciafino10m",
                7:"Lanciafino10mBianca", 8:"Lanciafino10mMarrone", 9:"Lanciamaggioredi10mBianca",10:"Lanciamaggioredi10mMarrone",11:"Motobarca",
                12:"Motopontonerettangolare", 13:"MotoscafoACTV", 14:"Mototopo", 15:"Patanella",16:"Polizia",17:"Raccoltarifiuti",18:"Sandoloaremi",
                19:"Sanpierota",20:"Topa",21:"VaporettoACTV", 22:"VigilidelFuoco", 23:"Water"}
    return dizionario[numero]


DATA_DIR = 'F:\\Università\\secondo anno\\primo semestre\\Caponnetto\\PMC 2021 Xibilia Caponetto\\sc5'
train_transform = T.Compose([T.Resize((240,240)),T.RandomHorizontalFlip(),T.ToTensor()])
train_data = datasets.ImageFolder(DATA_DIR, transform=train_transform)
print(train_data.classes)
print(torch.__version__)
trainloader = DataLoader(train_data, batch_size=32, shuffle=True)
for data in trainloader:
    print(data)
    break
x, y = data[0][0], data[1][0]

total = 0
counter_dict = {}
for label in train_data.classes:
    counter_dict[label] = 0

print(counter_dict)
for data in trainloader:
    xs, ys = data
    for y in ys:
        counter_dict[translating_data(int(y))] += 1

print(counter_dict)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(57600, 64)
