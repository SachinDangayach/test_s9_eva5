# Module to train CIFAR10 data

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.optim.lr_scheduler import StepLR

from tqdm import tqdm

def train(model, device, train_loader, optimizer, epoch, train_losses, train_acc):
  model.train()
  pbar = tqdm(train_loader)
  correct = 0
  processed = 0
  for batch_idx, (data, target) in enumerate(pbar):

    data, target = data.to(device), target.to(device)     # Get batch
    optimizer.zero_grad() # Set the gradients to zero before starting to do backpropragation
    y_pred = model(data)  # Predict

    loss = F.nll_loss(y_pred, target) # Calculate loss

    if m_num in {0,2,4}:
      # Calculate the MSE and L1
      l1 = calculate_l1_reg(model, lambda_l1)
      loss = loss + l1

    train_losses.append(loss) # Accumulate loss per batch

    # Backpropagation
    loss.backward()
    optimizer.step()

    # Update pbar-tqdm
    pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
    correct += pred.eq(target.view_as(pred)).sum().item()
    processed += len(data)

    pbar.set_description(desc= f'Loss={loss.item()} Batch_id={batch_idx} Train Accuracy={100*correct/processed:0.2f}')
    train_acc.append(100*correct/processed)

    return loss.item(), 100*correct/processed, train_losses, train_acc
