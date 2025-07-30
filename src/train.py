
import torch
import os
from torchvision import transforms
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
from cnn import default_cnn
from torch.utils.tensorboard import SummaryWriter


def train_model(train_data=None, 
                model=default_cnn(), 
                loss_module=nn.CrossEntropyLoss(), 
                optimizer=optim.Adam(default_cnn().parameters(), lr=1e-3), 
                epochs=120, 
                early_stop=10, 
                warmup=15, 
                min_val_loss=1000000,
                model_name=None):
    
    # NOVO: Inicializa o writer do TensorBoard. Os logs serão salvos na pasta 'runs/experiment_name'
    
    writer = SummaryWriter(f'runs/{model_name}')
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    cnt = 0
    global_step = 0 # NOVO: Contador global de batches para o TensorBoard

    # Loop sobre dataset de treino
    for epoch in range(epochs):
        # NOVO: Acumuladores para as métricas da época
        epoch_train_loss = 0.0
        epoch_val_loss = 0.0
        epoch_train_correct = 0
        epoch_train_total = 0
        epoch_val_correct = 0
        epoch_val_total = 0
        
        pbar = tqdm(train_data, desc=f"Epoch {epoch+1}/{epochs}", leave=True)
        
        for batch in pbar:
            model.train()
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)
            sz = int(0.8 * len(inputs))
            inputs_train, labels_train, inputs_val, val_labels = inputs[:sz], labels[:sz], inputs[sz:], labels[sz:]
            
            # --- Treino do mini-batch ---
            optimizer.zero_grad()
           
            outputs_train = model(inputs_train) 
            loss_train = loss_module(outputs_train, labels_train.long())
            loss_train.backward()
            optimizer.step()
            
            # NOVO: Cálculo da acurácia de treino do batch
            _, predicted_train = torch.max(outputs_train.data, 1)
            batch_train_total = labels_train.size(0)
            batch_train_correct = (predicted_train == labels_train).sum().item()
            batch_train_acc = batch_train_correct / batch_train_total
            
            # NOVO: Acumula métricas da época
            epoch_train_loss += loss_train.item()
            epoch_train_total += batch_train_total
            epoch_train_correct += batch_train_correct

            # --- Validação do mini-batch ---
            model.eval()
            with torch.no_grad():
                outputs_val = model(inputs_val)
                loss_val = loss_module(outputs_val, val_labels.long())
                if loss_val.item() < min_val_loss: min_val_loss = loss_val.item()

                # NOVO: Cálculo da acurácia de validação do batch
                _, predicted_val = torch.max(outputs_val.data, 1)
                batch_val_total = val_labels.size(0)
                batch_val_correct = (predicted_val == val_labels).sum().item()
                batch_val_acc = batch_val_correct / batch_val_total

                # NOVO: Acumula métricas da época
                epoch_val_loss += loss_val.item()
                epoch_val_total += batch_val_total
                epoch_val_correct += batch_val_correct
            
            # NOVO: Registra métricas por batch (eixo X = batches)
            writer.add_scalar('Loss/train_batch', loss_train.item(), global_step)
            writer.add_scalar('Accuracy/train_batch', batch_train_acc, global_step)
            writer.add_scalar('Loss/val_batch', loss_val.item(), global_step)
            writer.add_scalar('Accuracy/val_batch', batch_val_acc, global_step)
            global_step += 1

            # Atualiza barra de progresso
            pbar.set_postfix({
                "train_loss": f"{loss_train.item()} |",
                "val_loss": f"{loss_val.item()} |",
                "min_val_loss": f"{min_val_loss}"
            })

        # NOVO: Calcula médias da época e registra no TensorBoard (eixo X = épocas)
        avg_epoch_train_loss = epoch_train_loss / len(train_data)
        avg_epoch_val_loss = epoch_val_loss / len(train_data)
        avg_epoch_train_acc = epoch_train_correct / epoch_train_total
        avg_epoch_val_acc = epoch_val_correct / epoch_val_total
        
        writer.add_scalar('Loss/train_epoch', avg_epoch_train_loss, epoch)
        writer.add_scalar('Accuracy/train_epoch', avg_epoch_train_acc, epoch)
        writer.add_scalar('Loss/val_epoch', avg_epoch_val_loss, epoch)
        writer.add_scalar('Accuracy/val_epoch', avg_epoch_val_acc, epoch)

        # Lógica de early stopping (usando a última loss de validação do batch da época)
        if epoch > warmup and loss_val.item() > min_val_loss: 
            cnt += 1
        
        if epoch > warmup and cnt >= early_stop:
            print('Early stopping ativado!')
            break

        if avg_epoch_val_loss <= min_val_loss: # Salva o modelo baseado na loss média da época
            torch.save(model.state_dict(), f'model_checkpoint/model_{model_name}.pth')

    # NOVO: Fecha o writer para garantir que tudo foi salvo
    writer.close()
    print('Treinamento concluído!')