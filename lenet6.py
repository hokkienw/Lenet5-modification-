import torch

class LeNet6(torch.nn.Module):
  def __init__(self, activation='ReLu', pooling='Max', padd=1, in_chan=3):
    super(LeNet6, self).__init__()

    if activation == 'Tanh':
      self.activation_function = torch.nn.Tanh()
    elif activation == 'ReLu':
      self.activation_function  = torch.nn.ReLU()
    elif activation == 'Sigmoid':
      self.activation_function  = torch.nn.Sigmoid()
    else:
      raise NotImplementedError

    if pooling == 'Average':
      self.pooling_layer = torch.nn.AvgPool2d(kernel_size=2, stride=2)
    elif pooling == 'Max':
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

