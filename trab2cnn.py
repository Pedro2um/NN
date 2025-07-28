
import kagglehub
from pathlib import Path
import os
from PIL import Image
import torch
import torchvision.transforms.v2 as v2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, IterableDataset
import torch.optim as optim
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from pathlib import Path # Import Path
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torch.optim as optim

# Download latest version
ds_path = kagglehub.dataset_download("alsaniipe/chest-x-ray-image")
ds_path

data_path = os.path.join(ds_path,"Data/")
train_path = os.path.join(data_path, "train/")
test_path = os.path.join(data_path, "test/")


def load_img(path):
  # Le a imagem em diversos formatos e garante que a imagem tenha 3 canais
  img = Image.open(path).convert('RGB')
  # converte para um tensor do pytorch
  img = v2.functional.to_image(img)
  # garante que seja uma imagem de 8 bits reescalando os valores adequadamente
  img = v2.functional.to_dtype(img, dtype=torch.uint8, scale=True) # Remove this line
  return img # Return as PIL Image


class ChestXrayDataset(IterableDataset):
  def __init__(self, data_path, batch_size, transforms):
    self._data_path = data_path
    self._batch_size = batch_size # for each class
    self._transforms = transforms

  def __iter__(self):
    class_dirs = sorted([d for d in Path(self._data_path).iterdir() if d.is_dir()]) # Convert to Path
    class_iters = [iter(d.iterdir()) for d in class_dirs]

    # balanced batch
    while True:
      batch = []
      all_done = True


      for class_idx, class_iter in enumerate(class_iters):
        class_batch = []
        try:
          for _ in range(self._batch_size):
            path = next(class_iter)
            sample = self._preprocess(path)
            class_batch.append(sample)
          batch.extend(class_batch)
          all_done = False
        except StopIteration:
          continue # class folder ended

      if batch:
        yield batch
      if all_done:
        break

  def _preprocess(self, path):
    img = load_img(path)
    return (self._transforms(img), self._get_label_from_path(path))

  def _get_label_from_path(self, path):
    _path = str(path).lower() # Convert path to string for lower()
    if 'covid' in _path: return 0
    elif 'normal' in _path: return 1
    else: return 2  # pneumonia

from torchvision import transforms

train_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])
train_data = ChestXrayDataset(data_path=train_path, batch_size=128, transforms=train_transforms)
train_data_loader = DataLoader(dataset=train_data, batch_size=1, num_workers=0)


test_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])
test_data = ChestXrayDataset(data_path=test_path, batch_size=128, transforms=test_transforms)
test_data_loader = DataLoader(dataset=test_data, batch_size=1, num_workers=0)




def train_valid_split_idx(batch_size=None, n_classes=3, train_size=0.8):
  train = list()
  valid = list()
  if train_size > 1.0 or train_size <= 0:
    train_size = 0.8

  for _ in range(n_classes):
    i = 0
    j = i + (batch_size//n_classes)
    idx = list(range(i, j))
    np.random.shuffle(idx)
    _end = int(train_size * len(idx))
    train.extend(idx[:_end])
    valid.extend(idx[_end:])

  train = np.asarray(train)
  valid = np.asarray(valid)

  return train, valid


def save_checkpoint(epoch, checkpoint_dir, state_dict, save_every_nth_epoch=2):
  if (epoch + 1) % save_every_nth_epoch == 0:
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith('model_checkpoint') and f.endswith('.pth')]
    checkpoints = sorted(checkpoints, key=lambda f: os.path.getmtime(os.path.join(checkpoint_dir, f)))

    if len(checkpoints) >= 3:
      oldest = checkpoints[0]
      os.remove(os.path.join(checkpoint_dir, oldest))

    new_index = epoch + 1
    checkpoint_path = os.path.join(checkpoint_dir, f'model_checkpoint{new_index}.pth')
    torch.save(state_dict, checkpoint_path)

def train_model(model=None, optimizer=None, train_data_loader=None, loss_module=None, epochs=200, logging_dir='drive/MyDrive/Experimentos/runs/cnn'):
  model.train()
  device = next(model.parameters()).device
  #writer = SummaryWriter(logging_dir)

  for epoch in tqdm(range(epochs)):
    epoch_loss = []
    val_loss = []

    # OBS: validação por batch ao invés de ser por epoch (pior, sim, mas é um teste válido caso seu dataset seja IMENSO (mundo real) )

    for batch in train_data_loader:
      # split train and validation images
      #inputs, labels = map(list, zip(*batch)) # Remove this line
      inputs, labels = zip(*batch) # Unpack the batch into images and labels
      # Reshape inputs to flatten the first two dimensions (DataLoader batch and Dataset batch)
      inputs = torch.cat(inputs, dim=0).to(device) # Concatenate the tensors in the list along the batch dimension
      labels = torch.cat(labels, dim=0).to(device) # Concatenate the labels as well

      # inputs = torch.stack(inputs).numpy() # Remove this line
      # labels = np.array(labels) # Remove this line

      train_idx, val_idx = train_valid_split_idx(len(inputs)) # Use the total number of samples in the combined batch
      inputs_train, labels_train, inputs_val, val_labels  = inputs[train_idx], labels[train_idx], inputs[val_idx], labels[val_idx]

      ##########################################################################
      #training mini-batch
      optimizer.zero_grad()
      #outputs = model(inputs) # Remove this line
      outputs = model(inputs_train) # Use inputs_train
      loss = loss_module(outputs, labels_train.long()) # Use labels_train and convert to long
      loss.backward()
      optimizer.step()
      epoch_loss.append(loss.item())
      ##########################################################################
      # validation mini-batch
      with torch.no_grad():
        #outputs = model(inputs_val) # Remove this line
        outputs = model(inputs_val) # Use inputs_val
        loss = loss_module(outputs, val_labels.long()) # Use val_labels and convert to long
        val_loss.append(loss.item())

      save_checkpoint(epoch=epoch, checkpoint_dir='drive/MyDrive/Experimentos/NN/', state_dict=model.state_dict(), save_every_nth_epoch=10)
      print(f'epoch {epoch} train loss: {epoch_loss[-1]} val loss {val_loss[-1]}')



    #print(train_imgs.shape) # (153, 1, 3, 224, 224)
    #print(valid_imgs.shape) # (39, 1, 3, 224, 224)

# feature extractor
conv = nn.Sequential(
    nn.Conv2d(3, 6, 5), # Changed input channels from 1 to 3
    nn.ReLU(),
    nn.MaxPool2d(2, 2),
    nn.Conv2d(6, 16, 5),
    nn.ReLU(),
    nn.MaxPool2d(2, 2),
    nn.Flatten(),   # lembre-se do flatten!
    nn.Linear(16 * 53 * 53, 120), # Updated input size to 16 * 53 * 53
    nn.ReLU(),
    nn.Linear(120, 84)
)

# rede completa: feature extractor + classifier
net = nn.Sequential(
    conv,
    nn.ReLU(),
    nn.Linear(84, 3) # Changed output size to 3 for 3 classes
)



loss_module = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=1e-3)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)
train_model(model=net, optimizer=optimizer, train_data_loader=train_data_loader, loss_module=loss_module, epochs=3)

torch.save(net.state_dict(), 'drive/MyDrive/Experimentos/NN/model_initial.pth')