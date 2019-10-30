import pandas as pd 
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
import torch.utils.data
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

torch.manual_seed(1)

data_frame = pd.read_csv('0_3_min_labeled.csv')

input_size = 6
laneWidth = 12
#LSTM reads in one timestep at a time.
per_element = True

if per_element:
    lstm_input_size = 1
else:
    lstm_input_size = input_size

# size of hidden layers
h1 = 32
output_dim = 3
num_layers = 2
learning_rate = 1e-3
num_epochs = 6000
#dtype = torch.cuda.FloatTensor

#store data in 2d matrix
data = np.array(data_frame)
scaler = StandardScaler()
vehicleIDs = []
numbOfVehicleIDs = 0
f = []
y = []
for i in range(input_size):
	f.append([])
#get the different vehicle ID's
countin = 0
for i in range(len(data)):
	if data[i,4] not in vehicleIDs:
		vehicleIDs.append(data[i,4])
groundTruth = [0,0,0]
for i in range(len(data)):
	#if data[i,4] == vehicleIDs[j]:
	#	print(data[i,29])
	if data[i,4] == 720:
		y.append(data[i,29])
		if data[i,29] == 0:
			groundTruth[0] += 1
		elif data[i,29] == 1:
			groundTruth[1] += 1
		else:
			groundTruth[2] += 1
		if data[i,8] + data[i,13]/2 >= data[i,17] * laneWidth or data[i,8] - data[i,13]/2 <= (data[i,17] - 1) * (laneWidth):
			f[4].append(-1)
		else:
			f[4].append(data[i,17])
		f[5].append(data[i,8] - data[i,17] * laneWidth - laneWidth/2)
		f[0].append(data[i,16])
		f[1].append(data[i,15]-data[i,32])
		if data[i,24] == 0:
			f[2].append(0)
		else:
			tempTime = data[i,7]
			tempId = data[i,24]
			for j in range(-10,10):
				if data[i+j,7] == tempTime and data[i+j,24] == tempId:
					f[2].append(data[i,15]-data[i+j,15])
		f[3].append(data[i,31])
#print(countin)
#num_train = len(f[0])
print(groundTruth)

#second car
# groundTruth = [0,0,0]
# for i in range(len(data)):
# 	if data[i,4] == 0:
# 		print(data[i,29])
# 	if data[i,4] == 0:
# 		y.append(data[i,29])
# 		if data[i,29] == 0:
# 			groundTruth[0] += 1
# 		elif data[i,29] == 1:
# 			groundTruth[1] += 1
# 		else:
# 			groundTruth[2] += 1
# 		if data[i,8] + data[i,13]/2 >= data[i,17] * laneWidth or data[i,8] - data[i,13]/2 <= (data[i,17] - 1) * (laneWidth):
# 			f[4].append(-1)
# 		else:
# 			f[4].append(data[i,17])
# 		f[5].append(data[i,8] - data[i,17] * laneWidth - laneWidth/2)
# 		f[0].append(data[i,16])
# 		f[1].append(data[i,15]-data[i,32])
# 		if data[i,24] == 0:
# 			f[2].append(0)
# 		else:
# 			tempTime = data[i,7]
# 			tempId = data[i,24]
# 			for j in range(-10,10):
# 				if data[i+j,7] == tempTime and data[i+j,24] == tempId:
# 					f[2].append(data[i,15]-data[i+j,15])
# 		f[3].append(data[i,31])

num_train = len(f[0])
xtrain = np.asarray(f)
xtrain = scaler.fit_transform(xtrain)
print(xtrain)


X_train = torch.from_numpy(xtrain).type(torch.Tensor)

X_train = X_train.view([input_size, -1, 1])
X_train.requires_grad = True

#Arrange labels on one-hot encoded vectors
ytrain = np.asarray(y)
y_train = torch.from_numpy(ytrain).type(torch.LongTensor).view(-1)
print(y_train)
#trainloader=torch.utils.data.DataLoader(X_train, batch_size=100, shuffle=False, num_workers=8)

# Here we define our model as a class
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

model = LSTM(lstm_input_size, h1, batch_size=num_train, output_dim=output_dim, num_layers=num_layers)

#loss_fn = nn.MSELoss()
#loss_fn = nn.L1Loss()
loss_fn = nn.CrossEntropyLoss()
#loss_fn = torch.nn.KLDivLoss()
optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
#optimiser = torch.optim.SGD(model.parameters(), lr=learning_rate)

#####################
# Train model
#####################

hist = np.zeros(num_epochs)

print('Training...\n')

for t in range(num_epochs):
#	if t == 0:		
	model.hidden = model.init_hidden()

	y_pred = model(X_train)
	
	loss = loss_fn(y_pred, y_train)

	print("Epoch ", t, "\nMSE: ", loss.item())

	hist[t] = loss.item()

	optimiser. zero_grad()

	loss.backward()

	optimiser.step()


torch.save(model.state_dict(), 'net.pt')

print('\nFinished training.')
print(y_pred[0:60,:])