
from utils.download import download_data_and_create_annotations
from dataset import ChestXrayDataset
from cnn import default_transform, default_cnn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import pandas as pd
from torchvision.datasets import ImageFolder
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
import kagglehub
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report
import pandas as pd

from test import test_model
from train import train_model

DSL = True # se true, joga tudo na RAM usando joblib Parallel


def run_model(model, model_name):
    loss_module = torch.nn.CrossEntropyLoss()
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    if 'frozen' in model_name: 
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)# congela o extrator de features
    

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    train_model(train_data=train, model=model, loss_module=loss_module, optimizer=optimizer, epochs=120, early_stop=15, warmup=10, min_val_loss=1000000, model_name=model_name)
    
    preds, labels = test_model(model, test_dataset=test, target_names=test_data.classes)
    target_names = test_data.classes
    report_dict = classification_report(labels, preds, target_names=target_names, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    print(report_df)
    report_df.to_csv(f"classification_report_{model_name}.csv", index=True)

if __name__ == '__main__':

    # Dados e configuração de ambiente (se joga todas as imagens na RAM ou Não)
    ds_path = kagglehub.dataset_download("alsaniipe/chest-x-ray-image") + '/Data/'
    train_dir= 'train/'
    test_dir= 'test/'
    train_ds_path = os.path.join(ds_path, train_dir)
    test_ds_path = os.path.join(ds_path, test_dir)
    train_data = ImageFolder(train_ds_path)
    test_data = ImageFolder(test_ds_path)

   
    # padrão para o colab
    cache = False
    n_jobs = 1

    if DSL == True: # se estiver no DSL
        cache = True
        n_jobs = -1


    
    train_dataset = ChestXrayDataset(data=train_data, 
                                     transforms=default_transform(), 
                                     cache=cache, 
                                     n_jobs=n_jobs)
    
    test_dataset = ChestXrayDataset(data=test_data, 
                                    transforms=default_transform(), 
                                    cache=cache, 
                                    n_jobs=n_jobs)

    train = DataLoader(     dataset=train_dataset, 
                            batch_size=64, 
                            shuffle=True, 
                            num_workers=2)
    
    test = DataLoader(     dataset=test_dataset, 
                           batch_size=64, 
                           num_workers=2)


    # CNN 1 - customizada
    # Modelo e parâmetros de treinamento
    model1 = default_cnn()

    # CNN 2 - MobileNetV3-Large pré-treinada no ImageNet (fazer frozen do extrator de features)
    model2 = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1)
    model2.classifier[3] = torch.nn.Linear(model2.classifier[3].in_features, len(train_data.classes))

    # CNN 3 - MobileNetV3-Large pré-treinada no ImageNet (faze fine-tuning)
    model3 = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1)
    model3.classifier[3] = torch.nn.Linear(model3.classifier[3].in_features, len(train_data.classes))


    models = {'custom': model1, 'mobile_net_v3_large_frozen': model2, 'mobile_net_v3_large_fine-tuning': model3}

    for model_name, model in models.items():
        run_model(model=model, model_name=model_name)
    
 
    

    