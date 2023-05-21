import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch

from NN_config import Config
from HD_DataLoader import *

from NN_model import *


def train():
  # Configuration settings
  cfg = Config()

  # Load dataset
  dataset = {} #function_giving_sets(split='train') # call the data loader from the HD_DataLoader.py
  trainloader = DataLoader(dataset, batch_size=cfg.batch_size_train, shuffle=True, num_workers=4)

  # Initialize network
  model = Classifier(cfg)
  model.train()
  if cfg.enable_cuda:
    model = model.cuda()

  # Initialize optimizer
  optimizer = optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.lr_momentum, weight_decay=cfg.weight_decay)
  criterion = nn.CrossEntropyLoss()

  # Loop over inputs
  running_loss = 0.0
  i = 0
  
  print("Starting training...")
  
  for epoch in range(cfg.num_epochs):
    for i, data in enumerate(trainloader, 0):
      # get the inputs; data is a list of [inputs, labels]
      inputs, labels = data

      # zero the parameter gradients
      optimizer.zero_grad()

      # forward + backward + optimize
      outputs = model(inputs)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()

      # print statistics
      running_loss += loss.item()
      if i % 2000 == 1999:    # print every 2000 mini-batches
          print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
          running_loss = 0.0

  print("Finished training.")
  save_path = 'model.pth'
  torch.save(model.state_dict(), save_path)
  print("Saved trained model as {}.".format(save_path))

if __name__ == "__main__":
  train()
