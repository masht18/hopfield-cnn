# -*- coding: utf-8 -*-

import os
import random
import torch
import pickle
from torch import nn
from torch import optim
import torchvision
import torchvision.transforms as transforms

from torch.nn import Flatten, Linear, Module, Sequential
from torch.utils.data import DataLoader, Subset

from utils import *
from model import *


epochs = 100
batch_sz = 32
hopfield_neurons = 32
cnn_save_loc = 'CNN2.pth'
hopfield_save_loc = 'memory_hopfield.pth'
readout_save_loc = 'readout_hopfield.pth'
results_save_loc = 'CNN_only_loss.pkl'


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


trainset = torchvision.datasets.CIFAR10(root='data', train=True,
                                        download=False, transform=transform)

testset = torchvision.datasets.CIFAR10(root='data', train=False,
                                       download=False, transform=transform)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

selected_classes = [0, 1, 2, 3]
train_label_ref = generate_label_reference(trainset)
test_label_ref = generate_label_reference(testset)

test_indices = [indices for c in selected_classes for indices in test_label_ref[c]]
subset_test = Subset(testset, test_indices) 
subset_test_dataloader = DataLoader(subset_test, batch_size=batch_sz, shuffle=True)

train_indices = [indices for c in selected_classes for indices in train_label_ref[c]]
subset_train = Subset(trainset, train_indices) 
subset_train_dataloader = DataLoader(subset_train, batch_size=batch_sz, shuffle=True)

def train(dataloader):
    """
    Execute one training epoch.
    
    :param network: network instance to train
    :param optimiser: optimiser instance responsible for updating network parameters
    :param data_loader: data loader instance providing training data
    :return: tuple comprising training loss as well as accuracy
    """
    CNN.train()
    readout.train()
    losses, accuracies = [], []
    for data in dataloader:
        x, target = data
        #data, target = data.to(device=device), target.to(device=device)

        # Process data by Hopfield-based network.
        output = CNN(x)
        output = readout(output)

        # Update network parameters.
        optimizer.zero_grad()
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        # Compute performance measures of current model.
        _, predicted = torch.max(output.data, 1)
        #print(output)
        #print(predicted)
        accuracy = (torch.eq(predicted, target)).sum().item()/batch_sz
        accuracies.append(accuracy)
        losses.append(loss.detach().item())
    
    # Report progress of training procedure.
    return (sum(losses) / len(losses), sum(accuracies) / len(accuracies))

def train_hopfield():
    """
    Store exemplar patterns of each class to hopfield network
    """
    stored_patterns = []
    
    for ind in selected_classes:
        exemplar, target = trainset[random.choice(train_label_ref[ind]).item()]
        #data, target = data.to(device=device), target.to(device=device)

        # Process data by Hopfield-based network.
        with torch.no_grad():
            latents = CNN(exemplar.unsqueeze(0))
            latents = torch.sign(latents)
            hopfield.train_weights(latents)
            stored_patterns.append(latents)
            
    return stored_patterns
    
def evaluate_hopfield(dataloader, stored_patterns):
    
    not_recovered = 0
    misremembered = 0
    correct = 0
    all_latents = []
    all_predicted = []
    labels = []
    
    with torch.no_grad():
        for sample_data in dataloader:
            data, target = sample_data
            latents = CNN(data)
            latents = torch.sign(latents)
            predictions = hopfield.predict(latents)
            
            for i in range(predictions.shape[0]):
                if torch.equal(predictions[i], stored_patterns[target[i].item()]):
                    correct += 1
                elif bool([torch.equal(predictions[i], pattern) for pattern in stored_patterns]):
                    misremembered += 1
                    #print(predictions[i])
                else:
                    not_recovered += 1
                all_latents.append(latents[i])
                all_predicted.append(predictions[i])
                labels.append(target[i]) 
                
    return correct/4000, misremembered/4000, not_recovered/4000, all_latents, all_predicted, labels
                

def evaluate(dataloader):
    """
    Execute one training epoch.
    
    :param network: network instance to train
    :param optimiser: optimiser instance responsible for updating network parameters
    :param data_loader: data loader instance providing training data
    :return: tuple comprising training loss as well as accuracy
    """
    CNN.eval()
    readout.eval()
    accuracies = []
    with torch.no_grad():
        for sample_data in dataloader:
            data, target = sample_data
            #data, target = data.to(device=device), target.to(device=device)
    
            # Process data by Hopfield-based network.
            output = CNN(data)
            output = readout(output)
    
            # Compute performance measures of current model.
            _, predicted = torch.max(output.data, 1)
            accuracy = (torch.eq(predicted, target)).sum().item()/batch_sz
            accuracies.append(accuracy)
        
    # Report progress of training procedure.
    return sum(accuracies) / len(accuracies)


#~~~~~~~~~~~~~~TRAIN CNN~~~~~~~~~~~~~~~~~~~~~~~~~
CNN = CNN(output_sz=hopfield_neurons)
readout = nn.Linear(32, 4)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam([{'params': CNN.parameters(), 'lr': 0.001},
    {'params': readout.parameters(), 'lr': 0.001}])

if os.path.exists(cnn_save_loc):
    CNN.load_state_dict(torch.load(cnn_save_loc))
    print("Loading CNN")
    
else:
    trainlog = {'loss': [], 'train_acc': [], 'test_acc': []}
    
    for i in range(epochs):
        loss, train_acc = train(subset_train_dataloader)
        val_acc = evaluate(subset_test_dataloader)
        printlog = '| epoch {:3d} | loss {:5.4f} | train accuracy {:1.5f} | val accuracy {:1.5f}'.format(i, 
                                                                                    loss, 
                                                                                    train_acc,
                                                                                    val_acc)
        print(printlog)
        
        trainlog['loss'].append(loss)
        trainlog['train_acc'].append(train_acc)
        trainlog['test_acc'].append(val_acc)
        
        torch.save(CNN.state_dict(), cnn_save_loc)
        torch.save(readout.state_dict(), readout_save_loc)
        
        with open(results_save_loc, "wb") as f:
            pickle.dump(trainlog, f)
    

hopfield = Hopfield(hopfield_neurons)
stored_patterns = train_hopfield()
correct, misremembered, not_recovered, latents, predicted, targets = evaluate_hopfield(subset_test_dataloader, 
                                                          stored_patterns)
printlog = '| remembered: {:1.5f} | misremembered {:1.5f} | not remembered {:1.5f}'.format(correct,
                                                                                    misremembered,
                                                                                    not_recovered)
print(printlog)


