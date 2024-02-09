# -*- coding: utf-8 -*-

import torch
import numpy as np
import torchvision.datasets
import matplotlib.pyplot as plt

#модернизированная LeNet5
class LeNet5(torch.nn.Module):
  def __init__(self, activation='tanh',
                 pooling='avg',
                 conv_size=5,
                 use_batch_norm=False,
                 padd=0,in_chan=1):
    super(LeNet5, self).__init__()
    self.conv_size = conv_size
    self.use_batch_norm = use_batch_norm

    if activation == 'tanh':
      self.activation_function = torch.nn.Tanh()
    elif activation == 'relu':
      self.activation_function  = torch.nn.ReLU()
    elif activation == 'sigmoid':
      self.activation_function  = torch.nn.Sigmoid()
    else:
      raise NotImplementedError

    if pooling == 'avg':
      self.pooling_layer = torch.nn.AvgPool2d(kernel_size=2, stride=2)
    elif pooling == 'max':
      self.pooling_layer  = torch.nn.MaxPool2d(kernel_size=2, stride=2)
    else:
      raise NotImplementedError

    # первый сверточный слой

    if conv_size == 5:
      self.conv1 = torch.nn.Conv2d(
        in_channels=in_chan, out_channels=6, kernel_size=5, padding=padd)
    elif conv_size == 3:
      if padd == 2:
        padd =1
      self.conv1_1 = torch.nn.Conv2d(
        in_channels=in_chan, out_channels=6, kernel_size=3, padding=padd)
      self.conv1_2 = torch.nn.Conv2d(
        in_channels=6, out_channels=6, kernel_size=3, padding=padd)
    else:
      raise NotImplementedError


    self.act1 = self.activation_function
    self.bn1 = torch.nn.BatchNorm2d(num_features=6)
    self.pool1 = self.pooling_layer

    # второй сверточный слой

    if conv_size == 5:
      self.conv2 = self.conv2 = torch.nn.Conv2d(
                  in_channels=6, out_channels=16, kernel_size=5, padding=0)
    elif conv_size == 3:
      self.conv2_1 = torch.nn.Conv2d(
                  in_channels=6, out_channels=16, kernel_size=3, padding=0)
      self.conv2_2 = torch.nn.Conv2d(
                  in_channels=16, out_channels=16, kernel_size=3, padding=0)
    else:
      raise NotImplementedError

    self.act2 = self.activation_function
    self.bn2 = torch.nn.BatchNorm2d(num_features=16)
    self.pool2 = self.pooling_layer

    #fullyconnected слои
    self.fc1 = torch.nn.Linear(5 * 5 * 16, 120)
    self.fc1_act = self.activation_function
    self.fc2 = torch.nn.Linear(120, 84)
    self.fc2_act = self.activation_function
    self.fc3 = torch.nn.Linear(84, 10)

  def forward(self, x):
    if self.conv_size == 5:
      x = self.conv1(x)
    elif self.conv_size == 3:
      x = self.conv1_1(x)
      x = self.conv1_2(x)

    x = self.act1(x)

    if self.use_batch_norm:
      x = self.bn1(x)

    x = self.pool1(x)

    if self.conv_size == 5:
      x = self.conv2(x)
    elif self.conv_size == 3:
      x = self.conv2_1(x)
      x = self.conv2_2(x)

    x = self.act2(x)
    if self.use_batch_norm:
      x = self.bn2(x)

    x = self.pool2(x)

    x = x.view(x.size(0), x.size(1) * x.size(2) * x.size(3))

    x = self.fc1(x)
    x = self.fc1_act(x)
    x = self.fc2(x)
    x = self.fc2_act(x)
    x = self.fc3(x)
    return x


class CifarNet(torch.nn.Module):
  def __init__(self, activation='relu', pooling='max', padd=1, in_chan=3):
    super(CifarNet, self).__init__()

    if activation == 'tanh':
      self.activation_function = torch.nn.Tanh()
    elif activation == 'relu':
      self.activation_function  = torch.nn.ReLU()
    elif activation == 'sigmoid':
      self.activation_function  = torch.nn.Sigmoid()
    else:
      raise NotImplementedError

    if pooling == 'avg':
      self.pooling_layer = torch.nn.AvgPool2d(kernel_size=2, stride=2)
    elif pooling == 'max':
      self.pooling_layer  = torch.nn.MaxPool2d(kernel_size=2, stride=2)
    else:
      raise NotImplementedError

    self.batch0 = torch.nn.BatchNorm2d(num_features=in_chan)

    self.conv1 = torch.nn.Conv2d(in_chan, 16, 3, padding=padd)
    self.act1 = self.activation_function
    self.batch1 = torch.nn.BatchNorm2d(16)
    self.pool1 = self.pooling_layer

    self.conv2 = torch.nn.Conv2d(16, 32, 3, padding=1)
    self.act2 = self.activation_function
    self.batch2 = torch.nn.BatchNorm2d(32)
    self.pool2 = self.pooling_layer

    self.conv3 = torch.nn.Conv2d(32, 64, 3, padding=1)
    self.act3 = self.activation_function
    self.batch3 = torch.nn.BatchNorm2d(64)

    self.fc1   = torch.nn.Linear(8 * 8 * 64, 256)
    self.act4  = self.activation_function
    self.batch4 = torch.nn.BatchNorm1d(256)

    self.fc2   = torch.nn.Linear(256, 64)
    self.act5  = self.activation_function
    self.batch5 = torch.nn.BatchNorm1d(64)

    self.fc3   = torch.nn.Linear(64, 10)

  def forward(self, x):
    x = self.batch0(x)
    x = self.conv1(x)
    x = self.act1(x)
    x = self.batch1(x)
    x = self.pool1(x)

    x = self.conv2(x)
    x = self.act2(x)
    x = self.batch2(x)
    x = self.pool2(x)

    x = self.conv3(x)
    x = self.act3(x)
    x = self.batch3(x)

    x = x.view(x.size(0), x.size(1)*x.size(2)*x.size(3))

    x = self.fc1(x)
    x = self.act4(x)
    x = self.batch4(x)

    x = self.fc2(x)
    x = self.act5(x)
    x = self.batch5(x)

    x = self.fc3(x)

    return x



class Parse():
  def __init__(self, d_type):
    self.d_type = d_type
    self.train_data = 0
    self.test_data = 0
    self.x_train = 0
    self.y_train = 0
    self.x_test = 0
    self.y_test = 0

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

    elif self.d_type == "CIFAR":
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


class Treiner_Tester():
  def __init__(self, net, number_of_epochs,
               parse_instance,
               loss='Cross', gpu=True,
               batch_s=100, opt='Adam',
               lerning_rate=3.0e-3):

    self.net = net
    self.batch_size = batch_s
    self.epochs = number_of_epochs
    self.x_train = parse_instance.x_train
    self.y_train = parse_instance.y_train
    self.x_test = parse_instance.x_test
    self.y_test = parse_instance.y_test
    self.test_accuracy_history = []
    self.test_loss_history = []

    if gpu:
      self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    self.net = self.net.to(self.device)
    if loss == 'Cross':
      self.loss = torch.nn.CrossEntropyLoss()
    elif loss == 'MSE':
      self.loss = torch.nn.MSELoss()

    if opt == 'Adam':
      self.optimizer = torch.optim.Adam(net.parameters(), lr=lerning_rate)
    elif opt == 'SGD':
      self.optimizer = torch.optim.SGD(net.parameters(), lr=lerning_rate, momentum=0.9)


  def Train(self):
    self.x_test = self.x_test.to(self.device)
    self.y_test = self.y_test.to(self.device)

    for epoch in range(self.epochs):
          order = np.random.permutation(len(self.x_train))
          for start in range(0, len(self.x_train), self.batch_size):
              self.optimizer.zero_grad()
              self.net.train()

              batch_indexes = order[start:start+self.batch_size]

              X_batch = self.x_train[batch_indexes].to(self.device)
              y_batch = self.y_train[batch_indexes].to(self.device)

              preds = self.net.forward(X_batch)

              loss_value = self.loss(preds, y_batch)
              loss_value.backward()

              self.optimizer.step()
              # print(f'Epoch {epoch + 1}/{self.epochs}, Loss: {loss_value.item()}')

          self.net.eval()
          test_preds = self.net.forward(self.x_test)
          self.test_loss_history.append(self.loss(test_preds, self.y_test).data.cpu())
          accuracy = (test_preds.argmax(dim=1) == self.y_test).float().mean().data.cpu()
          self.test_accuracy_history.append(accuracy)
    return self.test_accuracy_history, self.test_loss_history




if __name__ == "__main__":

  parse_instance = Parse("MNIST")
  parse_instance.ParseData()

  accuracies = {}
  losses = {}

# tanh
  trainer_tester_instance = Treiner_Tester(LeNet5(activation='tanh', pooling='avg',
                                                  conv_size=5,use_batch_norm=False, padd=2,
                                                  in_chan=1), 30, parse_instance)
  accuracies['tanh'], losses['tanh'] = \
   trainer_tester_instance.Train()


# relu_5
  trainer_tester_instance = Treiner_Tester(LeNet5(activation='relu', pooling='avg',
                                                  conv_size=5,use_batch_norm=False, padd=2,
                                                  in_chan=1), 30, parse_instance)
  accuracies['relu_5'], losses['relu_5'] = \
   trainer_tester_instance.Train()


# relu_3
  trainer_tester_instance = Treiner_Tester(LeNet5(activation='relu', pooling='avg',
                                                  conv_size=3,use_batch_norm=False, padd=2,
                                                  in_chan=1), 30, parse_instance)
  accuracies['relu_3'], losses['relu_3'] = \
   trainer_tester_instance.Train()


# relu_3_maxpooling
  trainer_tester_instance = Treiner_Tester(LeNet5(activation='relu', pooling='max',
                                                  conv_size=3,use_batch_norm=False, padd=2,
                                                  in_chan=1), 30, parse_instance)

  accuracies['relu_3_max'], losses['relu_3_max'] = \
   trainer_tester_instance.Train()


# relu_3_maxpooling + batch
  trainer_tester_instance = Treiner_Tester(LeNet5(activation='relu', pooling='max',
                                                  conv_size=3,use_batch_norm=True, padd=2,
                                                  in_chan=1), 30, parse_instance)

  accuracies['relu_3_max_batch'], losses['relu_3_max_batch'] = \
   trainer_tester_instance.Train()

#CifarNet

  trainer_tester_instance = Treiner_Tester(CifarNet(padd = 3, in_chan = 1), 30, parse_instance)

  accuracies['cifar'], losses['cifar'] = \
   trainer_tester_instance.Train()




  for experiment_id in accuracies.keys():
      plt.plot(accuracies[experiment_id], label=experiment_id)
  plt.legend()
  plt.title('Validation Accuracy')

  # for experiment_id in losses.keys():
  #   plt.plot(losses[experiment_id], label=experiment_id)
  # plt.legend()
  # plt.title('Validation Loss');
