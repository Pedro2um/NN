
import torch
from tqdm import tqdm
from sklearn.metrics import classification_report
import pandas as pd

def test_model(model=None, 
               test_dataset=None, 
               target_names=None):
    
# Avaliação no conjunto de teste
    model.eval()
    all_preds = []
    all_labels = []
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_dataset, desc="Testando"):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

   
    #target_names = test_data.classes
    
    return all_preds, all_labels
    

    
    