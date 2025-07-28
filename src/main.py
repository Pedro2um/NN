
from utils.download import download_data_and_create_annotations
from dataset import ChestXrayDataset
from cnn import default_transform
from torch.utils.data import DataLoader
import os
import pandas as pd
from torchvision.datasets import ImageFolder
import kagglehub

if __name__ == '__main__':

    ds_path = kagglehub.dataset_download("alsaniipe/chest-x-ray-image") + '/Data/'
    train_dir= 'train/'
    test_dir= 'test/'
    train_ds_path = os.path.join(ds_path, train_dir)
    test_ds_path = os.path.join(ds_path, test_dir)
    train_data = ImageFolder(train_ds_path)
    test_data = ImageFolder(test_ds_path)

    # cache = False and n_jobs = 1 no colab
    # cache = True and n_jobs = -1 no DSL
    train_dataset = ChestXrayDataset(train_data, transforms=default_transform(), cache=False, n_jobs=1)
    test_dataset = ChestXrayDataset(test_data, transforms=default_transform(), cache=False, n_jobs=1)

    train_data = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
    test_dataset = DataLoader(test_dataset, batch_size=64, num_workers=2)



    for batch in train_data:

        print(batch)
        break