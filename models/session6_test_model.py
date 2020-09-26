# Module to train

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

def test(model, device, test_loader):
  model.eval()
  test_loss = 0
  correct = 0
  with torch.no_grad():
      for data, target in test_loader:
          data, target = data.to(device), target.to(device)
          output = model(data)
          test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
          pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
          correct += pred.eq(target.view_as(pred)).sum().item()

  test_loss /= len(test_loader.dataset)
  test_losses.append(test_loss)

  print('\nTest set: Average loss: {:.4f}, Test Accuracy: {}/{} ({:.2f}%)\n'.format(
      test_loss, correct, len(test_loader.dataset),
      100. * correct / len(test_loader.dataset)))

  test_acc.append(100. * correct / len(test_loader.dataset))

  return test_loss, 100. * correct / len(test_loader.dataset), test_losses, test_acc

def record_max_acc(max_acc):
  f = open(base_folder+acc_recorder_file, "w")
  f.write(str(max_acc))
  f.close()
