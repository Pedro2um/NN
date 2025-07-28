import multiprocessing
from torch.utils.data import Dataset
from utils.image import load_img
from joblib import Parallel, delayed
import torch
import os

class ChestXrayDataset(Dataset):
  def __init__(self, data, transforms, cache, n_jobs=-1):
    self._data = data
    self._transforms = transforms
    self._cache = cache
    self._cached_imgs = {}

    # joga tudo na ram
    if cache:
      num_workers = multiprocessing.cpu_count() if n_jobs == -1 else n_jobs

      def load_and_return(idx):
          path, _ = self._data.samples[idx]
          return idx, load_img(path)

      results = Parallel(n_jobs=num_workers)(
          delayed(load_and_return)(idx) for idx in range(len(self._data))
      )

      self._cached_imgs = dict(results)

  def __len__(self):
    return len(self._data)

  def __getitem__(self, idx):
    path, label = self._data.samples[idx]
    if self._cache and idx in self._cached_imgs:
      img = self._cached_imgs[idx]
    else:
      img = load_img(path)

    if self._transforms:
      img = self._transforms(img)

    return img, torch.tensor(label, dtype=torch.long)