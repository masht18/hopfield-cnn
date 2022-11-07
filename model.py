import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class CNN(nn.Module):
    def __init__(self, input_channel=3, hidden1=6, hidden2=16, hidden3=32, output_sz=32):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channel, hidden1, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(hidden1, hidden2, 5)
        self.conv3 = nn.Conv2d(hidden2, hidden3, 3)
        self.fc1 = nn.Linear(hidden3 * 3 * 3, 120)
        self.fc2 = nn.Linear(120, output_sz)
        #self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        #x = self.fc3(x)
        return x


class Hopfield(nn.Module):
    def __init__(self, num_neurons, retrieval_iter=100, threshold=0):
        super().__init__()
        self.W = torch.zeros((num_neurons, num_neurons))
        self.retrieval_iter = retrieval_iter
        self.threshold = 0
        
    def train_weights(self, data):     
        # Loop through all data samples.
        # :param data (b x 32 (num_neurons))
        for dd in range(data.shape[0]):
    
            # Data cross-product gives neural hopfield update rule.
            w_update = data[dd].T * data[dd]
            
            # Sum all pattern cross-products.
            self.W = self.W + w_update
    
        # Hopfield nets are a form of RNNs, albeit without self-connections, and so, we need to make sure that the 
        # diagonal elements of the final weight matrix are zero. 
        self.W.fill_diagonal_(0)
        
    def predict(self, data):
        
        final_s = torch.zeros_like(data)
        
        for dd in range(data.shape[0]):
            final_s[dd] = self.retrieval(data[dd])
            
        return final_s
    
    # This function computes the Hopfield nets' energy.
    def energy(self, s):
        # x: [b x N] data vector
        # w: [N x N] hopfield net weight matrix.
        energy = torch.mm(torch.mm(-s, self.W), s.T)
        return energy
    
    def retrieval(self, s):

        #e = self.energy(s)
        s = s.unsqueeze(0)
        
        for i in range(self.retrieval_iter):
            s_new = torch.sign(torch.mm(s, self.W) - self.threshold)
            # Compute new state energy
            #e_new = self.energy(s)
            
            # s is converged
            if torch.equal(s, s_new):
                 return s
            # Update energy
            s = s_new
            
        return s
        