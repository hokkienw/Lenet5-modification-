#модернизированная LeNet5
import torch

class LeNet5(torch.nn.Module):
  def __init__(self, activation='Tanh',
                 pooling='Average',
                 conv_size=5,
                 use_batch_norm=False,
                 padd=0,in_chan=1):
    super(LeNet5, self).__init__()
    self.conv_size = conv_size
    self.use_batch_norm = use_batch_norm

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
