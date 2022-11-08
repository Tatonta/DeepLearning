import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torchvision.datasets as Datasetslel
FILE_PATH = 'F:\\Università\\secondo anno\\primo semestre\\Caponnetto\\PMC 2021 Xibilia Caponetto\\sc5'
def translating_data(numero):
    dizionario={0:"Alilaguna", 1:"Ambulanza", 2:"Barchino", 3:"Cacciapesca", 4:"Caorlina", 5:"Gondola", 6:"Lanciafino10m",
                7:"Lanciafino10mBianca", 8:"Lanciafino10mMarrone", 9:"Lanciamaggioredi10mBianca",10:"Lanciamaggioredi10mMarrone",11:"Motobarca",
                12:"Motopontonerettangolare", 13:"MotoscafoACTV", 14:"Mototopo", 15:"Patanella",16:"Polizia",17:"Raccoltarifiuti",18:"Sandoloaremi",
                19:"Sanpierota",20:"Topa",21:"VaporettoACTV", 22:"VigilidelFuoco", 23:"Water"}
    return dizionario[numero]


dataset = Datasetslel.ImageFolder(FILE_PATH)
print(dataset)
X_train, X_test = train_test_split(dataset, test_size = 0.2, random_state = 1234)

data_translation = dataset.class_to_idx
counter_dict = {}
for i,label in enumerate(data_translation):
    counter_dict[i] = label

for data in X_train:
    xs, ys = data


print(counter_dict)