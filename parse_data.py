import torch
import torchvision.datasets

class Parse():
  def __init__(self, d_type):
    self.d_type = d_type

  def ParseData(self):
    if self.d_type == "MNIST":
      self.train_data = torchvision.datasets.MNIST('./', download=True, train=True)
      self.test_data = torchvision.datasets.MNIST('./', download=True, train=False)
      self.x_train = self.train_data.train_data
      self.y_train = self.train_data.train_labels
      self.x_test = self.test_data.test_data
      self.y_test = self.test_data.test_labels
      self.x_train = self.x_train.unsqueeze(1).float()
      self.x_test = self.x_test.unsqueeze(1).float()

    elif self.d_type == "CIFAR10":
      self.train_data = torchvision.datasets.CIFAR10('./', download=True, train=True)
      self.test_data = torchvision.datasets.CIFAR10('./', download=True, train=False)
      self.x_train = torch.FloatTensor(self.train_data.data)
      self.y_train = torch.LongTensor(self.train_data.targets)
      self.x_test = torch.FloatTensor(self.test_data.data)
      self.y_test = torch.LongTensor(self.test_data.targets)
      self.x_train /= 255.
      self.x_test /= 255.
      self.x_train = self.x_train.permute(0, 3, 1, 2)
      self.x_test = self.x_test.permute(0, 3, 1, 2)
    else:
      raise NotImplementedError