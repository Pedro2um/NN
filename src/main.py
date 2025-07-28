
from utils.download import download_data_and_create_annotations
from dataset import ChestXrayDataset
from cnn import default_transform, default_cnn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import pandas as pd
from torchvision.datasets import ImageFolder
import kagglehub
import torch
import numpy as np
from tqdm import tqdm

DSL = True # se true, joga tudo na RAM usando joblib Parallel

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

    cache = False
    n_jobs = 1

    if DSL == True:
        cache = True
        n_jobs = -1

    train_dataset = ChestXrayDataset(train_data, transforms=default_transform(), cache=cache, n_jobs=n_jobs)
    test_dataset = ChestXrayDataset(test_data, transforms=default_transform(), cache=cache, n_jobs=n_jobs)

    train_data = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
    test_dataset = DataLoader(test_dataset, batch_size=64, num_workers=2)

    model = default_cnn()
    


    if os.path.exists('model_checkpoint/model.pth'):
        print('tem o modelo')
        state_dict = torch.load('model_checkpoint/model.pth')
        model.load_state_dict(state_dict)

    
    loss_module = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    epochs = 120
    min_val_loss = 1000000
    cnt = 0
    edging = 15 # early stoping
    warmup = 10

    # loop sobre dataset de treino
    for epoch in tqdm(range(epochs)):
        epoch_loss = []
        val_loss = []
        
        # divide cada batch em treino e validação
        for batch in train_data:
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)
            sz = int(0.8 * len(inputs))
            inputs_train, labels_train, inputs_val, val_labels  = inputs[:sz], labels[:sz], inputs[sz:], labels[sz:]
            
            ##########################################################################
            #treino mini-batch
            optimizer.zero_grad()
           
            outputs = model(inputs_train) 
            loss = loss_module(outputs, labels_train.long())
            loss.backward()
            optimizer.step()
            epoch_loss.append(loss.item())
            ##########################################################################
            # validação mini-batch
            with torch.no_grad():
                outputs = model(inputs_val)
                loss = loss_module(outputs, val_labels.long())
                val_loss.append(loss.item())
                if loss.item() < min_val_loss: min_val_loss = loss.item()

            print(f'epoch {epoch} train loss: {epoch_loss[-1]} val loss {val_loss[-1]}')

        if epoch > warmup and loss.item() > min_val_loss: 
            cnt+=1
        
        if epoch > warmup and cnt >= edging:
            print('acabou')
            break

        if val_loss[-1] <= min_val_loss:
            torch.save(model.state_dict(), 'model_checkpoint/model.pth')
        


    