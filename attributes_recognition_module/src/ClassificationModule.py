import torch
import torch.nn as nn
import torch.nn.functional as F

class ClassificationModule(nn.Module):
  def __init__(self, input_size, num_classes):
    super(ClassificationModule, self).__init__()
    self.fc1 = nn.Linear(input_size,512) # first dense layer
    self.fc2 = nn.Linear(512, 256) #Second dense layer
    self.fc3 = nn.Linear(256, 128) # Third dense layer
    self.fc4 = nn.Linear(128, num_classes) # Output Layer
    self.num_classes = num_classes

    self.dropout = nn.Dropout(p=0.5)

  def forward(self, x):
    x = F.relu(self.fc1(x))
    x = self.dropout(x)
    x = F.relu(self.fc2(x))
    x = self.dropout(x)
    x = F.relu(self.fc3(x))
    x = self.dropout(x)
    x = F.relu(self.fc4(x))
    #print(" Stampa Gigi "+str(self.fc4.parameters()))
    if self.num_classes == 1:
      return torch.sigmoid(x) # output: num_classes
    else:
      return F.softmax(x, dim=1)  # output: num_classes
    