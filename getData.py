import pandas as pd
import numpy as np
import pickle

input_size = 6
laneWidth = 12

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

csvData = pd.read_csv('0_3_min_labeled.csv')
data = np.array(csvData)

vehicleIDs = []
for i in range(len(data)):
	if data[i,4] not in vehicleIDs:
		vehicleIDs.append(data[i,4])

f = []
for i in range(9):
	f.append([])

vehicle = []
for i in range(len(vehicleIDs)):
	f = []
	for k in range(13):
		f.append([])
	for j in range(len(data)):
		if data[j,4] == vehicleIDs[i]:
			f[0].append(data[j,7])
			f[1].append(data[j,8])
			f[2].append(data[j,13])
			f[3].append(data[j,15])
			f[4].append(data[j,16])
			if data[j,8] + data[j,13]/2 >= data[j,17] * laneWidth or data[j,8] - data[j,13]/2 <= (data[j,17] - 1) * (laneWidth):
				f[5].append(-1)
			else:
				f[5].append(data[j,17])
			f[6].append(data[j,8] - data[j,17] * laneWidth - laneWidth/2)
			f[7].append(data[j,24])
			f[8].append(data[j,29])
			f[9].append(data[j,31])
			f[10].append(data[j,32])
			if data[j,24] == 0:
				f[11].append(data[j,15])
			else:
				tempTime = data[j,7]
				tempId = data[j,24]
				if j > 10 and j < (len(data) - 10):
					for c in range(-10,10):
						if data[j+c,7] == tempTime and data[j+c,24] == tempId:
							f[11].append(data[j,15]-data[j+c,15])
			f[12].append(data[j,15] - data[j,32])
	vehicle.append(Vehicle(input_size, laneWidth, vehicleIDs[i], f[0], f[1], f[2], f[3], f[4], f[5], f[6], f[7], f[8], f[9], f[10], f[11], f[12]))

vehicleClass = open('vehicle.obj', 'wb')
pickle.dump(vehicle, vehicleClass)