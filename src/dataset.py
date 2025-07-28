from torch.utils.data import Dataset
from utils.image import load_img
import torch
import os

class ChestXrayDataset(Dataset):
  def __init__(self, annotation_data, ds_path, transforms):
    self._annotation_data = annotation_data
    self._transforms = transforms
    self._ds_path = ds_path

  def __len__(self):
    return len(self._data)

  def __getitem__(self, idx):
    #print(self._annotation_data[idx, :])
    img_path, label = self._annotation_data[idx, :]
    img = load_img(os.path.join(self._ds_path,img_path))
    tensor = self._transforms(img)
    label = torch.tensor(label)
    return tensor, label