import pandas as pd 
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(1)

data_frame = pd.read_csv('1_min_labeled.csv')

data = np.array(data_frame)
a = np.array
b = np.array
a = data[:,2:18]
b = np.hstack((a,data[:,24:28]))
xData = np.hstack((b,data[:,30:33])).astype(np.float)

c0 = 0
c1 = 0
c2 = 0

for i in range(len(data)):
	if data[i,29] == 0:
		c0 += 1
	elif data[i,29] == 1:
		c1 += 1
	elif data[i,29] == 2:
		c2 += 1

print(c0, c1, c2)