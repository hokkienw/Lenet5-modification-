import torch
import numpy as np

class Treiner_Tester():
  def __init__(self, net, number_of_epochs,
               parse_instance,
               loss='Cross Entropy', gpu="GPU",
               batch_s=100, opt='Adam',
               lerning_rate=3.0e-3, d_type="MNIST"):

    self.net = net
    self.d_type = d_type
    self.batch_size = batch_s
    self.epochs = number_of_epochs
    self.x_train = parse_instance.x_train
    self.y_train = parse_instance.y_train
    self.x_test = parse_instance.x_test
    # print(self.x_test.shape)
    self.y_test = parse_instance.y_test
    self.test_accuracy_history = []
    self.test_loss_history = []
    self.device = torch.device('cpu')

    if gpu == "GPU":
      self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    self.net = self.net.to(self.device)
    if loss == 'Cross Entropy':
      self.loss = torch.nn.CrossEntropyLoss()
    else:
      self.loss = torch.nn.MSELoss()

    if opt == 'Adam':
      self.optimizer = torch.optim.Adam(net.parameters(), lr=lerning_rate)
    else:
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

          self.net.eval()
          test_preds = self.net.forward(self.x_test)
          self.test_loss_history.append((self.loss(test_preds, self.y_test)).data.cpu())
          accuracy = (test_preds.argmax(dim=1) == self.y_test).float().mean().data.cpu()
          self.test_accuracy_history.append(accuracy)
    return self.test_accuracy_history, self.test_loss_history
  
  def ManTest(self, data):
    if (self.d_type == "MNIST"):
      test_preds = self.net.forward(data)
      predicted_class = torch.argmax(test_preds, dim=1).item()
    else:
      pred_list =  ["airplane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
      test_preds = self.net.forward(data) 
      predicted_class_ind = torch.argmax(test_preds, dim=1).item()
      predicted_class = pred_list[predicted_class_ind]
    return predicted_class
