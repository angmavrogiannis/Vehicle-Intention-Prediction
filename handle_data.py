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
#print(data_frame.values)
n = len(data_frame)

num_datapoints = n
input_size = 17
test_size = 0.2
#num_train = int((1-test_size) * num_datapoints)
num_train = 11500

#LSTM reads in one timestep at a time.
per_element = True

if per_element:
    lstm_input_size = 1
else:
    lstm_input_size = input_size



# size of hidden layers
h1 = 32
output_dim = 3
num_layers = 1
learning_rate = 1e-3
num_epochs = 100
#dtype = torch.cuda.FloatTensor

#store data in 2d matrix
data = np.array(data_frame)
a = np.array
b = np.array
a = data[:,8:18]
b = np.hstack((a,data[:,24:28]))
xData = np.hstack((b,data[:,30:33])).astype(np.float)
scaler = StandardScaler()
#xtrain = xData[0:1000,:]
#xtrain = np.array
x0 = np.empty([6780, 17], dtype=float) #need to do this with parameters, not with scalars
x1 = np.empty([1587, 17], dtype=float)
x2 = np.empty([296058, 17], dtype=float)
c0 = 0
c1 = 0
c2 = 0

for i in range(len(data)):
	if data[i,29] == 0:
		x0[c0,:] = xData[i,:]
		c0 += 1
	elif data[i,29] == 1:
		x1[c1,:] = xData[i,:]
		c1 += 1
	elif data[i,29] == 2:
		x2[c2,:] = xData[i,:]
		c2 += 1

x01 = np.vstack((x0[0:5000,:], x1[0:1500,:]))
xtrain = np.vstack((x01, x2[0:5000,:]))
#xtrain = x01
#xtrain = xData[0:15000,:]
xtrain = scaler.fit_transform(xtrain)
print(xtrain)


X_train = torch.from_numpy(xtrain).type(torch.Tensor)
X_train = X_train.view([input_size, -1, 1])
X_train.requires_grad = True

#Arrange labels on one-hot encoded vectors
yData = np.zeros((len(data), 3))
for i in range(len(data)):
	if data[i,29] == 0:
		yData[i,0] = 1
	elif data[i,29] == 1:
		yData[i,1] = 1
	elif data[i,29] == 2:
		yData[i,2] = 1
ytrain = np.zeros([11500, 3])
for i in range(11500):
	if i < 5000:
		ytrain[i,0] = 1
	elif i < 6500:
		ytrain[i,1] = 1
	elif i < 11500:
		ytrain[i,2] = 1


#ytrain = yData[0:15000,:]

#print(ytrain[2000:2010,:])
y_train = torch.from_numpy(ytrain).type(torch.Tensor).view(-1)

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
        return y_pred.view(-1)

model = LSTM(lstm_input_size, h1, batch_size=num_train, output_dim=output_dim, num_layers=num_layers)

loss_fn = nn.MSELoss()
#loss_fn = nn.L1Loss()
#loss_fn = nn.CrossEntropyLoss()
#loss_fn = torch.nn.KLDivLoss()
optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)

#####################
# Train model
#####################

hist = np.zeros(num_epochs)

print('Training...\n')

for t in range(num_epochs):
#	if t == 0:		
#		model.hidden = model.init_hidden()

	y_pred = model(X_train)

	loss = loss_fn(y_train, y_pred)

	print("Epoch ", t, "\nMSE: ", loss.item())
	hist[t] = loss.item()

	optimiser. zero_grad()

	loss.backward()

	optimiser.step()

#print(hist)

#torch.save(model.state_dict(), "network.pth")


#plt.plot(y_pred.detach().numpy(), label="Preds")
#plt.plot(y_train.detach().numpy(), label="Data")
#plt.legend()
#plt.show()


torch.save(model.state_dict(), 'net.pt')

#show figure loss-epochs
#plt.plot(hist, label="Training loss")
#plt.legend()
#plt.show()

print('\nFinished training.')
print(y_pred[0:20])
print(y_pred[15500:15521])
print(y_pred[34400:34421])
count = 0
i = 0
cp0 = 0
cp1 = 0
cp2 = 0
while i < 34500:
	if y_pred[i] >= y_pred[i+1] and y_pred[i] >= y_pred[i+2]:
		y_pred[i] = 1.
		y_pred[i+1] = 0.
		y_pred[i+2] = 0.
	elif y_pred[i+1] >= y_pred[i] and y_pred[i+1] >= y_pred[i+2]:
		y_pred[i+1] = 1.
		y_pred[i] = 0.
		y_pred[i+2] = 0.
	elif y_pred[i+2] >= y_pred[i] and y_pred[i+2] >= y_pred[i+1]:
		y_pred[i+2] = 1.
		y_pred[i] = 0.
		y_pred[i+1] = 0.
	if y_train[i] == 1:
		if y_pred[i] == 1:
			cp0 += 1
	elif y_train[i+1] == 1:
		if y_pred[i+1] == 1:
			cp1 += 1
	elif y_train[i+2] == 1:
		if y_pred[i+2] == 1:
			cp2 += 1
	else:
		print('somethings wrong')
		print(y_train[i],y_train[i+1],y_train[i+2])
	i += 3
	count += 1

print(y_pred[0:20])
print(y_pred[15500:15521])
print(y_pred[34400:34421])
print(cp0, cp1, cp2)