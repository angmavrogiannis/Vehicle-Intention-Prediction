# Vehicle-Intention-Prediction
This repository sums up the work I did in the Intelligent Control Lab at Carnegie Mellon University, Robotics Institute for the spring semester of 2018.

I got real-world vehicle trajectory data from NGSIM (https://ops.fhwa.dot.gov/trafficanalysistools/ngsim.htm) and labeled them based on the action-maneuver the vehicles are executing at a certain timestep.

- 0 for turning left
- 1 for turning right
- 2 for following current lane

Then I implemented a Long Short-Term Memory (LSTM) network in order to predict drivers' maneuver intention for a short time interval in the future.
I used PyTorch for training and Cross-Entropy loss for performing multiclass classification.
