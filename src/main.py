
from cnn import default_cnn, mobile_net
from dataset import ChestXrayDataset
from transforms import test_transform, vannila_transform
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import pandas as pd
from torchvision.datasets import ImageFolder
import kagglehub
import torch
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

from test import test_model
from train import train_model

DSL = True # se true, joga tudo na RAM usando joblib Parallel, precisa de uns 50gb só pras imagens, com 60gb de RAM roda tranquilo ;)


def run_model(model, model_name):

    path = os.path.join("results", f"{model_name}")
    if os.path.exists(path) == False:
        os.makedirs(path)


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
    report_df.to_csv(os.path.join(path, f"classification_report_{model_name}.csv"), index=True)

    cm = confusion_matrix(labels, preds)
    cm_df = pd.DataFrame(cm, index=target_names, columns=target_names)
    cm_df.to_csv(os.path.join(path, f"confusion_matrix_{model_name}.csv"), index=True)

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


    # somente resize
    train_dataset = ChestXrayDataset(data=train_data,
                                     transforms=vannila_transform(),  
                                     cache=cache, 
                                     n_jobs=n_jobs)
    
    test_dataset = ChestXrayDataset(data=test_data, 
                                    transforms=test_transform(), 
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
    model1 = default_cnn()
    # CNN 2 - MobileNetV3-Large pré-treinada no ImageNet (fazer frozen do extrator de features)
    model2 = mobile_net(len(train_data.classes))
    # CNN 3 - MobileNetV3-Large pré-treinada no ImageNet (faze fine-tuning)
    model3 = mobile_net(len(train_data.classes))
    
    models = {'custom': model1, 'mobile_net_v3_large_frozen': model2, 'mobile_net_v3_large_fine-tuning': model3}

    for model_name, model in models.items():
        run_model(model=model, model_name=model_name)
    
 
    

    