# -*- coding: utf-8 -*-

# import torch
import numpy as np
# import torchvision.datasets
import matplotlib.pyplot as plt
from lenet5 import LeNet5
from lenet6 import LeNet6
from resnet import ResNet
from parse_data import Parse
from trainer_tester import Treiner_Tester

class Model():
  def __init__(self, activation='Tanh',
                 pooling='Average',
                 conv_size=5,
                 use_batch_norm=False,
                net = "LeNet6", number_of_epochs=1,
                parse_data = "MNIST",
                loss='Cross Entropy', gpu="GPU",
                batch_s=100, opt='Adam',
                lerning_rate=3.0e-3):
    super(Model, self).__init__()
    self.parser = Parse(parse_data)
    self.parser.ParseData()
    if parse_data == "MNIST":
      self.padd = 2
      self.in_chan = 1
    else:
      self.padd = 0
      self.in_chan = 3
    if net == "LeNet6":
      self.net_ = LeNet6(activation, pooling, self.padd + 1, self.in_chan)
    elif net == "LeNet5":
      self.net_ = LeNet5(activation, pooling, conv_size, use_batch_norm, self.padd, self.in_chan)
    else:
      self.net_ = ResNet(padd=(self.padd+ 1), in_chan=self.in_chan)
    self.trainer = Treiner_Tester(self.net_, number_of_epochs, self.parser, loss, gpu, batch_s, opt, lerning_rate, parse_data)
      
    
  def Train(self):
    return self.trainer.Train()
  
  def ManTest(self, data):
    return self.trainer.ManTest(data)
