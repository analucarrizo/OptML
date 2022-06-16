# -*- coding: utf-8 -*-
"""helpers.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1_m0p7oEiPBpiRGEpil84-VaOAs2bKsB5
"""

import sklearn
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
import torchvision
from torch.nn import functional as F

import numpy as np
import time 

import itertools
from tqdm.notebook import tqdm

import torchvision.datasets as datasets

# !pip install CosineAnnealingWithRestartsLR
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import LinearLR
from torch.optim.lr_scheduler import ExponentialLR

def train(model, data_loader, criterion, optimizer, device, view_every = 500):
    running_loss = 0.0

    for i, data in enumerate(data_loader, 0):
        
        # get the input data
        inputs = data[0].to(device)
        labels = data[1].to(device)
        
        # zero the gradients
        optimizer.zero_grad()

        # forward + backward + optimizte
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % view_every == view_every-1: 
            print('Batch {:d} / {:d}: loss = {:.4f}'.format(i+1,len(data_loader),running_loss/view_every))
            running_loss = 0.0

def test(model, data_loader, device):
    correct = 0.0
    total = 0.0
        
    with torch.no_grad():
      for data in data_loader:
        images = data[0].to(device)
        labels = data[1].to(device)
        
        # compute outputs by running images through the network
        outputs = model(images)
        
        # choosing class with highest probability          
        pdf = F.softmax(outputs, dim=1) 
        predicted = pdf.argmax(dim=1)     

        correct += (predicted ==  labels).sum()
        total += len(predicted)
    return correct/total

def run_best_model(mnist_train, mnist_test, model, config, device):
  train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=64, shuffle=True, pin_memory=torch.cuda.is_available())
  test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=64, shuffle=True, pin_memory = torch.cuda.is_available())
  
  start_time = time.time()
  nb_epochs = config["epochs"]
  nb_folds = config["folds"]
  accs = [0]*nb_epochs

  for fold in range(nb_folds):
    model.to(device)

    batch_size = config["batch_size"]
    criterion = config["criterion"]()
    optimizer = config["optimizer"](model.parameters(), lr = config["lr"])
    scheduler = config["scheduler"](optimizer, **config["scheduler_parameters"])
    
    print(f"------ FOLD {fold + 1} -------")
    
    for epoch in range(nb_epochs):
      train(model, train_loader, criterion, optimizer, device)
      acc = test(model, test_loader, device)

      accs[epoch] += acc

      scheduler.step()

      print(f"Epoch {epoch}: lr = {scheduler.get_lr()}, test accuracy {acc:.4}")

  for i in range(len(accs)):
    accs[i] /= nb_folds

  print(f"----- {((time.time() - start_time)/60):.4} minutes")
  return accs