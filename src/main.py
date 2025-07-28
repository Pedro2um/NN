
from utils.download import download_data_and_create_annotations
from dataset import ChestXrayDataset
from cnn import default_transform
from torch.utils.data import DataLoader
import os
import pandas as pd

if __name__ == '__main__':

    train_file_name = 'train'
    test_file_name = 'test'

    ds_path = download_data_and_create_annotations('./', train_file_name, test_file_name)

    train_ds_path = os.path.join(ds_path, train_file_name)
    test_ds_path = os.path.join(ds_path, test_file_name)

    train_df = pd.read_csv(f'{train_file_name}.csv').to_numpy()
    
    test_df = pd.read_csv(f'{test_file_name}.csv').to_numpy()
    # até aqui tudo certo

    train_data = ChestXrayDataset(train_df, ds_path=train_ds_path, transforms=default_transform())
    test_data = ChestXrayDataset(test_df, ds_path=test_ds_path, transforms=default_transform())


    for batch in train_data:
        print(batch)
        break