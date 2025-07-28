
import torch
import os


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
    