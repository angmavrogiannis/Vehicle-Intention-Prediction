import pandas as pd 
import torch
import pickle
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

input_size = 6
num_train = 200

#LSTM reads in one timestep at a time.
#per_element = True
lstm_input_size = input_size


# size of hidden layers
h1 = 500
output_dim = 3
num_layers = 1
learning_rate = 1e-3
num_epochs = 2000

class Vehicle:
    def __init__(self, input_size, laneWidth, vehicleID, time, x, width, velocity, acceleration, laneID, xDev, preceding, intention, velx, velMean, velRel, velTraffic):
        self.laneWidth = laneWidth
        self.id = vehicleID #column 4
        self.x = x #column 8
        self.laneID = laneID #column 17 or -1 if occupyin two lanes at a time
        self.width = width #column 13
        self.velocity = velocity #column 15
        self.acceleration = acceleration #column 16
        self.preceding = preceding #column 24
        self.intention = intention #column 29
        self.velx = velx #column 31
        self.velMean = velMean #column 32
        self.time = time #column 7
        self.velRel = velRel #velocity relative to the front car
        self.xDev = xDev #lateral deviation from the center of current lane
        self.velTraffic = velTraffic #speed of ego vehicle, relative to the traffic flow
        self.features = []

        for i in range(input_size):
            self.features.append([])

        self.features[4] = self.laneID
        self.features[5] = self.xDev
        self.features[0] = self.acceleration
        self.features[1] = self.velTraffic
        self.features[2] = self.velRel
        self.features[3] = self.velx
        self.labels = self.intention

filehandler = open('vehicle.obj', 'rb')
vehicle = pickle.load(filehandler)

# trainCars = []
# for i in range(len(vehicle)):
#     for j in range(len(vehicle[i].labels)):
#         if vehicle[i].labels[j] != 2:
#             trainCars.append(i)
#             break

# count = 0
# label = 0
# for i in range(len(trainCars)):
#     carIndex = trainCars[i]
#     print('\n')
#     for j in range(len(vehicle[carIndex].labels)):
#         if j == 0:
#             print('Car ', carIndex, ' with ID: ', vehicle[carIndex].id, '\n')
#             count = 1
#             label = vehicle[carIndex].labels[j]
#         else: 
#             if vehicle[carIndex].labels[j] != vehicle[carIndex].labels[j-1]:
#                 print(count, 'behaviors of ', label, ', ')
#                 count = 1
#                 label = vehicle[carIndex].labels[j]
#             else:
#                 count += 1
#                 if j == len(vehicle[carIndex].labels) - 1:
#                     print(count, 'behaviors of ', label, ', ')

# counter = 0
# for i in range(20):
#     counter += len(vehicle[i].features[0])

# Here we define our model as a class
class NeuralNet(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim=3):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)


    def forward(self, input):

        out = self.fc1(input)
        out = self.relu(out)
        out = self.fc2(out)
        return out

model = NeuralNet(lstm_input_size, h1, output_dim=output_dim)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

#####################
# Train model
#####################

print('Training...\n')

lim1 = [310, 220, 300, 450, 200, 550, 30, 50, 120, 300, 400, 173, 250]
lim2 = [510, 420, 500, 650, 400, 750, 230, 250, 320, 500, 600, 373, 450]
trainIDs = [7, 9, 14, 18, 22, 33, 47, 128, 132, 145, 197, 212, 215]

for i in range(13):
    ind = trainIDs[i]
    x_train = np.asarray(vehicle[ind].features)
    print(x_train.shape)
    print(x_train)
    #x_train = scaler.fit_transform(x_train)
    x_train = x_train[:,lim1[i]:lim2[i]]
    X_train = torch.from_numpy(x_train).type(torch.Tensor)
    X_train = X_train.view([input_size, -1, 1])
    X_train = X_train.view(-1, input_size)
    print(X_train.shape)
    y_train = np.asarray(vehicle[ind].labels)
    y_train = y_train[lim1[i]:lim2[i]]
    y_train = torch.LongTensor(y_train)
    for t in range(num_epochs):
        y_pred = model(X_train)
        
        loss = loss_fn(y_pred, y_train)
        if t % 10 == 0:
            print("Epoch ", t, "\nMSE: ", loss.item())

        optimizer. zero_grad()

        loss.backward()

        optimizer.step()

torch.save(model.state_dict(), 'net.pt')
print('\nFinished training.')