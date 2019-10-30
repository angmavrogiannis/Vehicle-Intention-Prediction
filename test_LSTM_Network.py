import pandas as pd 
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

input_size = 6
hidden_dim = 6
output_dim = 3
num_layers = 2
#num_test = 1005
num_test = 200
per_element = True
training_size = 13 * 200
laneWidth = 12
test_vehicle = 212
lim1 = 173
lim2 = 373

if per_element:
    lstm_input_size = 1
else:
    lstm_input_size = input_size


class Vehicle:
	def __init__(self, input_size, laneWidth, vehicleID, time, x, width, velocity, acceleration, laneID, xDev, preceding, intention, velx, velMean, velRel, velTraffic):
		self.laneWidth = laneWidth
		self.id = vehicleID #column 4
		self.x = x #column 8
		self.laneID = laneID #column 17 or -1 if occupying two lanes at a time
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

groundTruth = [0,0,0]
for i in range(lim1, lim2):
	if vehicle[test_vehicle].labels[i] == 0:
		groundTruth[0] += 1
	elif vehicle[test_vehicle].labels[i] == 1:
		groundTruth[1] += 1
	else:
		groundTruth[2] += 1
		
print(groundTruth)
xtest = np.asarray(vehicle[test_vehicle].features)
#xtest = scaler.fit_transform(xtest)
xtest = xtest[:,lim1:lim2]

X_test = torch.from_numpy(xtest).type(torch.Tensor)

X_test = X_test.view([input_size, -1, 1])
X_test.requires_grad = True

#Arrange labels on one-hot encoded vectors
ytest = vehicle[test_vehicle].labels
ytest = np.asarray(ytest)
ytest = ytest[lim1:lim2]
y_test = torch.LongTensor(ytest).view(-1)
print(y_test)

yeval = np.zeros((num_test,3))
y_eval = torch.from_numpy(yeval).type(torch.Tensor)
for i in range(0, 200):
	if y_test[i] == 0:
		y_eval[i,0] = 1
	elif y_test[i] == 1:
		y_eval[i,1] = 1
	else:
		y_eval[i,2] = 1

c0 = groundTruth[0]
c1 = groundTruth[1]
c2 = groundTruth[2]

# print('Training data:\n\n',c2,' lane following examples(', c2/training_size*100,'%)\n\n', c0,
#  ' lane changing(0) examples(', c0/training_size*100, '%)\n', c1, 'lane changing(1) examples(', 
#  c1/training_size*100, '%)\n\n', c0+c1, 'total lane changing examples(', (c0+c1)/training_size*100, '%.')

class LSTM(nn.Module):

    def __init__(self, input_dim, hidden_dim, batch_size, output_dim,
                    num_layers):
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layers = num_layers

        # Define the LSTM layer
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers)

        # Define the output layer
        self.linear = nn.Linear(self.hidden_dim, output_dim)

    def init_hidden(self):
        # This is what we'll initialise our hidden state as
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

model = LSTM(lstm_input_size, hidden_dim, batch_size=num_test, output_dim=output_dim, num_layers=num_layers)
model.load_state_dict(torch.load('lstm_network.pt'))

print('Testing...')
with torch.no_grad():
	outputs = model(X_test)

print(outputs)

ct0 = 0
ct1 = 0
ct2 = 0
#predictions* when ground_truth=* 
predictions0 = [0, 0, 0]
predictions1 = [0, 0, 0]
predictions2 = [0, 0, 0]

counter = 0
i = 0

while i < len(outputs):
	if outputs[i,0] >= outputs[i,1] and outputs[i,0] >= outputs[i,2]:
		outputs[i,0] = 1
		outputs[i,1] = 0
		outputs[i,2] = 0
	elif outputs[i,1] >= outputs[i,0] and outputs[i,1] >= outputs[i,2]:
		outputs[i,1] = 1
		outputs[i,0] = 0
		outputs[i,2] = 0
	else:
		outputs[i,2] = 1
		outputs[i,0] = 0
		outputs[i,1] = 0
	if outputs[i,0] == y_eval[i,0] and outputs[i,1] == y_eval[i,1] and outputs[i,2] == y_eval[i,2]:
		counter += 1
	if y_eval[i,0] == 1:
		ct0 += 1
		if outputs[i,0] == 1:
			predictions0[0] += 1
		elif outputs[i,1] == 1:
			predictions0[1] += 1
		else:
			predictions0[2] += 1
	if y_eval[i,1] == 1:
		ct1 += 1
		if outputs[i,0] == 1:
			predictions1[0] += 1
		elif outputs[i,1] == 1:
			predictions1[1] += 1
		else:
			predictions1[2] += 1
	if y_eval[i,2] == 1:
		ct2 += 1
		if outputs[i,0] == 1:
			predictions2[0] += 1
		elif outputs[i,1] == 1:
			predictions2[1] += 1
		else:
			predictions2[2] += 1
	i += 1

if ct0 == 0:
	cct0 = 1
else:
	cct0 = ct0
if ct1 == 0:
	cct1 = 1
else:
	cct1 = ct1
if ct2 == 0:
	cct2 = 1
else:
	cct2 = ct2

print('Testing data:\n\n',ct2,' lane following examples(', ct2/num_test*100,'%)\n\n', ct0,
 ' lane changing(0) examples(', ct0/num_test*100, '%)\n', ct1, 'lane changing(1) examples(', 
 ct1/num_test*100, '%)\n\n', ct0+ct1, 'total lane changing examples(', (ct0+ct1)/num_test*100, '%.')
print('When y = 2 (lane following), we have:\n', predictions2[2], ' correct predictions (', predictions2[2]/cct2*100, '%)\n', 
	predictions2[0], ' wrong predictions for lane change(0) (', predictions2[0]/cct2*100, '%)\n', predictions2[1], 
	' wrong predictions for lange change(1) (', predictions2[1]/cct2*100, '%)\n Total: ', (predictions2[0]+predictions2[1])/cct2*100, 
	'% wrong predictions.')
print('When y = 0 (lane changing), we have:\n', predictions0[0], ' correct predictions (', predictions0[0]/cct0*100, '%)\n',
	predictions0[1], ' wrong predictions for lane change(1) (', predictions0[1]/cct0*100, '%)\n', predictions0[2], 
	' wrong predictions for lane following (2) (', predictions0[2]/cct0*100, '%)\n Total: ', (predictions0[1]+predictions0[2])/cct0*100, 
	'% wrong predictions.')
print('When y = 1 (lane changing), we have:\n', predictions1[1], ' correct predictions (', predictions1[1]/cct1*100, '%)\n',
	predictions1[0], ' wrong predictions for lane change(0) (', predictions1[0]/cct1*100, '%)\n', predictions1[2], 
	' wrong predictions for lane following (2) (', predictions1[2]/cct1*100, '%)\n Total: ', (predictions1[0]+predictions1[2])/cct1*100, 
	'% wrong predictions.\n')
print(counter, "correct predictions out of ", num_test, "total testing examples. Accuracy: ", counter/num_test*100, "%.")