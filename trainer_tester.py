import torch
import numpy as np

class Treiner_Tester():
  def __init__(self, net, number_of_epochs,
               parse_instance,
               loss='Cross Entropy', gpu="GPU",
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

    if gpu == "GPU":
      self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    self.net = self.net.to(self.device)
    if loss == 'Cross Entropy':
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
          self.test_loss_history.append((self.loss(test_preds, self.y_test)).data.cpu())
          accuracy = (test_preds.argmax(dim=1) == self.y_test).float().mean().data.cpu()
          self.test_accuracy_history.append(accuracy)
    return self.test_accuracy_history, self.test_loss_history
  
  def ManTest(self, data):
    # data_tensor = torch.tensor(data, dtype=torch.float32)
    # data_tensor = data_tensor.unsqueeze(0)
    test_preds = self.net.forward(data)
    return test_preds.argmax(dim=1)
