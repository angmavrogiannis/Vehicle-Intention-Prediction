import pandas as pd 
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

input_size = 17
h1 = 500
output_dim = 3
num_layers = 5
num_test = 160000
training_size = 16000

data_frame = pd.read_csv('0_3_min_labeled.csv')

data = np.array(data_frame)
a = np.array
b = np.array
a = data[:,8:18]
b = np.hstack((a,data[:,24:28]))
xData = np.hstack((b,data[:,30:33])).astype(np.float)

yData = np.zeros((len(data), 1))
for i in range(len(data)):
	if data[i,29] == 0:
		yData[i] = 0
	elif data[i,29] == 1:
		yData[i] = 1
	elif data[i,29] == 2:
		yData[i] = 2
scaler = StandardScaler()
xtest = xData[0:160000,:]

xtest = scaler.fit_transform(xtest)
X_test = torch.from_numpy(xtest).type(torch.Tensor)

ytest = yData[0:160000,:]
y_test = torch.from_numpy(ytest).type(torch.Tensor).view(-1)

c0 = 0
c1 = 0
c2 = 0
for i in range(training_size):
	if data[i,29] == 0:
		c0 += 1
	elif data[i,29] == 1:
		c1 += 1
	elif data[i,29] == 2:
		c2 += 1

#c0 = 5000
#c1 = 1000
#c2 = 10000

print('Training data:\n\n',c2,' lane following examples(', c2/training_size*100,'%)\n\n', c0,
 ' lane changing(0) examples(', c0/training_size*100, '%)\n', c1, 'lane changing(1) examples(', 
 c1/training_size*100, '%)\n\n', c0+c1, 'total lane changing examples(', (c0+c1)/training_size*100, '%).')

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

model = NeuralNet(input_size, h1, output_dim=output_dim)
model.load_state_dict(torch.load('feedForwardNet2.pt'))

print('Testing...')
with torch.no_grad():
	outputs = model(X_test)

ct0 = 0
ct1 = 0
ct2 = 0
#predictions* when ground_truth=* 
predictions0 = [0, 0, 0]
predictions1 = [0, 0, 0]
predictions2 = [0, 0, 0]

counter = 0
i = 0
pred = np.zeros((num_test, 1))

while i < len(outputs):
	if outputs[i,0] >= outputs[i,1] and outputs[i,0] >= outputs[i,2]:
		pred[i] = 0
	elif outputs[i,1] >= outputs[i,0] and outputs[i,1] >= outputs[i,2]:
		pred[i] = 1
	elif outputs[i,2] >= outputs[i,0] and outputs[i,2] >= outputs[i,1]:
		pred[i] = 2
	if y_test[i] == 0:
		ct0 += 1
		if pred[i] == 0:
			predictions0[0] += 1
			counter += 1
		elif pred[i] == 1:
			predictions0[1] += 1
		else:
			predictions0[2] += 1
	if y_test[i] == 1:
		ct1 += 1
		if pred[i] == 0:
			predictions1[0] += 1
		elif pred[i] == 1:
			counter += 1
			predictions1[1] += 1
		else:
			predictions1[2] += 1
	if y_test[i] == 2:
		ct2 += 1
		if pred[i] == 0:
			predictions2[0] += 1
		elif pred[i] == 1:
			predictions2[1] += 1
		else:
			counter += 1
			predictions2[2] += 1
	i += 1

print('Testing data:\n\n',ct2,' lane following examples(', ct2/num_test*100,'%)\n\n', ct0,
 ' lane changing(0) examples(', ct0/num_test*100, '%)\n', ct1, 'lane changing(1) examples(', 
 ct1/num_test*100, '%)\n\n', ct0+ct1, 'total lane changing examples(', (ct0+ct1)/num_test*100, '%.')
print('When y = 2 (lane following), we have:\n', predictions2[2], ' correct predictions (', predictions2[2]/ct2*100, '%)\n', 
	predictions2[0], ' wrong predictions for lane change(0) (', predictions2[0]/ct2*100, '%)\n', predictions2[1], 
	' wrong predictions for lange change(1) (', predictions2[1]/ct2*100, '%)\n Total: ', (predictions2[0]+predictions2[1])/ct2*100, 
	'% wrong predictions.')
print('When y = 0 (lane changing), we have:\n', predictions0[0], ' correct predictions (', predictions0[0]/ct0*100, '%)\n',
	predictions0[1], ' wrong predictions for lane change(1) (', predictions0[1]/ct0*100, '%)\n', predictions0[2], 
	' wrong predictions for lane following (2) (', predictions0[2]/ct0*100, '%)\n Total: ', (predictions0[1]+predictions0[2])/ct0*100, 
	'% wrong predictions.')
print('When y = 1 (lane changing), we have:\n', predictions1[1], ' correct predictions (', predictions1[1]/ct1*100, '%)\n',
	predictions1[0], ' wrong predictions for lane change(0) (', predictions1[0]/ct1*100, '%)\n', predictions1[2], 
	' wrong predictions for lane following (2) (', predictions1[2]/ct1*100, '%)\n Total: ', (predictions1[0]+predictions1[2])/ct1*100, 
	'% wrong predictions.\n')
print(counter, "correct predictions out of ", num_test, "total testing examples. Accuracy: ", counter/num_test*100, "%.")