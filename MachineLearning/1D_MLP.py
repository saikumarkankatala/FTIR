import numpy as np
import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix

#!pip install -Uqq ipdb
#import ipdb
#%pdb on

class Dataset_FTIR(Dataset):
  def __init__(self,x,y):
    self.x = torch.tensor(x).float()
    #normalize to [0,1]
    self.x = self.x / self.x.max()
    self.y = torch.tensor(y).long()
  def __len__(self):
    return len(self.x)
  def __getitem__(self, ix):
    return self.x[ix], self.y[ix]

data_path = 'Full_Data_Raw.csv'
data = pd.read_csv(data_path)

X = data.iloc[:, :-1].values  # Features
y = data.iloc[:, -1].values  # Labels

skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)

accuracies_folds = []

for train_index, test_index in skf.split(X, y):
  print('\nStarting the next fold...')

  # Splitting the data
  X_train, X_test = X[train_index], X[test_index]
  y_train, y_test = y[train_index], y[test_index]

  # Create dataset objects
  train_dataset = Dataset_FTIR(X_train, y_train)
  test_dataset = Dataset_FTIR(X_test, y_test)

  # Create DataLoader
  train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
  test_loader = DataLoader(test_dataset, batch_size=32)

  if 1:
    model = nn.Sequential(
        nn.Linear(2100, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 4)
    )
  else:
      model = nn.Sequential(
      nn.Linear(2100, 1028),
      nn.ReLU(),
      nn.Linear(1028, 512),
      nn.ReLU(),
      nn.Linear(512, 64),
      nn.ReLU(),
      nn.Linear(64, 4)
  )

  #summary(model, torch.zeros(1,2102))

  #print(model)

  # train the model
  loss_fn = nn.CrossEntropyLoss()  # Cross-entropy loss for multiclass classification
  optimizer = optim.Adam(model.parameters(), lr=0.00001)

  n_epochs = 100
  loss_history = []

  for epoch in range(n_epochs):
    for ix, iy in train_loader:

      #print('ix:', ix.shape, ' iy:', iy.shape)
      #print('epoch ', epoch, ' of ', n_epochs)
      #ipdb.set_trace(context=6)

      optimizer.zero_grad()

      loss_value = loss_fn(model(ix), iy)

      loss_value.backward()

      # update weights
      optimizer.step()

    loss_history.append(loss_value.detach().numpy())

    #print(f'Finished epoch {epoch}, latest loss {loss_value}')

  plt.figure(0)
  plt.plot(loss_history)
  plt.xlabel('epochs')
  plt.ylabel('loss value')
  plt.yscale("log")

  # compute accuracy (no_grad is optional)
  with torch.no_grad():
    X_test ,  y_test = test_dataset[:]
    y_pred = model(X_test)

    ce = loss_fn(y_pred, y_test)

    # generate lables from probabilities
    y_predLabel = torch.argmax(y_pred, 1)

    #acc = ( y_predLabel == y_test).float().mean()
    #print(f"Epoch {epoch} validation: Cross-entropy={float(ce)}, Accuracy={float(acc)}")
    correct = np.sum(np.array(y_predLabel) == np.array(y_test))
    total = len(y_test)
    fold_accuracy = 100 * correct / total
    print(f"Accuracy={float(fold_accuracy)}")
    accuracies_folds.append(fold_accuracy)

    cm = confusion_matrix(y_test, y_predLabel)
    print("Confusion Matrix :")
    print(cm)


# Plotting the accuracies
folds = range(1, len(accuracies_folds) + 1)
plt.figure(1)
plt.plot(folds, accuracies_folds, marker='o', linestyle='-', color='b')
plt.title('Accuracy per Fold')
plt.xlabel('Fold')
plt.ylabel('Accuracy (%)')
plt.xticks(folds)
plt.grid(True)
plt.show()