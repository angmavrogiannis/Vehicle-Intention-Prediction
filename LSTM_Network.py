import numpy as np
import pickle
import torch
import torch.nn as nn
import pandas as pd 
from sklearn.preprocessing import StandardScaler
import csv

input_size = 6
hidden_dim = 6
output_dim = 3
num_layers = 2
learning_rate = 1e-3
num_train = 200
num_epochs = 1000

#LSTM reads in one timestep at a time.
per_element = True

if per_element:
    lstm_input_size = 1
else:
    lstm_input_size = input_size

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

scaler = StandardScaler()
filehandler = open('vehicle.obj', 'rb')
vehicle = pickle.load(filehandler)
trainCars = []
for i in range(len(vehicle)):
	for j in range(len(vehicle[i].labels)):
		if vehicle[i].labels[j] != 2:
			trainCars.append(i)
			break

count = 0
label = 0
for i in range(len(trainCars)):
	carIndex = trainCars[i]
	print('\n')
	for j in range(len(vehicle[carIndex].labels)):
		if j == 0:
			print('Car ', carIndex, ' with ID: ', vehicle[carIndex].id, '\n')
			count = 1
			label = vehicle[carIndex].labels[j]
		else: 
			if vehicle[carIndex].labels[j] != vehicle[carIndex].labels[j-1]:
				print(count, 'behaviors of ', label, ', ')
				count = 1
				label = vehicle[carIndex].labels[j]
			else:
				count += 1
				if j == len(vehicle[carIndex].labels) - 1:
					print(count, 'behaviors of ', label, ', ')
print(trainCars)
print(vehicle[2].labels)
# c0 = []
# c1 = []
# c2 = []
# ids =[]
# for i in range(580):
# 	c0.append(0)
# 	c1.append(0)
# 	c2.append(0)
# 	ids.append(vehicle[i].id)
# 	for j in range(len(vehicle[i].labels)):
# 		if vehicle[i].labels[j] == 0:
# 			c0[i] += 1
# 		elif vehicle[i].labels[j] == 1:
# 			c1[i] += 1
# 		else:
# 			c2[i] += 1
# carData = zip(ids, c0, c1, c2)
# with open('carData.csv', 'w') as outfile:
# 	for entries in carData:
# 		outfile.write(str(entries))
# 		outfile.write('\n')
# carDataSize = []
# for i in range(580):
# 	carDataSize.append(len(vehicle[i].labels))
# print(min(carDataSize))

counter = 0
for i in range(20):
	counter += len(vehicle[i].features[0])
#print(counter)

class LSTM(nn.Module):

    def __init__(self, input_dim, hidden_dim, batch_size, output_dim,
                    num_layers):
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layers = num_layers

        #LSTM layer
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers)

        #Output layer
        self.linear = nn.Linear(self.hidden_dim, output_dim)

    def init_hidden(self):
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim),
                torch.zeros(self.num_layers, self.batch_size, self.hidden_dim))

    def forward(self, input):
        # Forward pass through LSTM layer
        # shape of lstm_out: [input_size, batch_size, hidden_dim]
        # shape of self.hidden: (a, b), where a and b both 
        # have shape (num_layers, batch_size, hidden_dim).
        lstm_out, self.hidden = self.lstm(input.view(len(input), self.batch_size, -1))
        
        # Only take the output from the final timestep
        # Can pass on the entirety of lstm_out to the next layer if it is a seq2seq prediction
        y_pred = self.linear(lstm_out[-1].view(self.batch_size, -1))
        return y_pred

model = LSTM(lstm_input_size, hidden_dim, batch_size=num_train, output_dim=output_dim, num_layers=num_layers)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

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
	print(X_train)
	y_train = np.asarray(vehicle[ind].labels)
	y_train = y_train[lim1[i]:lim2[i]]
	y_train = torch.LongTensor(y_train)
	for t in range(num_epochs):
		if t == 0:
			model.hidden = model.init_hidden()
		y_pred = model(X_train)
		
		loss = loss_fn(y_pred, y_train)
		if t % 10 == 0:
			print("Epoch ", t, "\nMSE: ", loss.item())

		optimizer. zero_grad()

		loss.backward()

		optimizer.step()


torch.save(model.state_dict(), 'lstm_network.pt')

print('\nFinished training.')
print(y_pred[0:60,:])