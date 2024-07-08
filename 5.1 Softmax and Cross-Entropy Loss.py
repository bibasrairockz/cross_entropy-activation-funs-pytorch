import torch
import torch.nn as nn
import numpy as np

# softmax with numpy
X= np.array([2, 1, 0.1])
def softmax(x):
    return np.exp(x)/np.sum(np.exp(x), axis= 0)
print(f"Softmax numpy: {softmax(X)}")

# softmax with pytorch
X= torch.tensor([2, 1, 0.1])
print(f"Softmax pytorch: {torch.softmax(X, dim= 0)}")

# cross entropy with numpy
Y= np.array([1, 0, 0])
Y_good= np.array([0.7, 0.2, 0.1])
Y_bad= np.array([0.1, 0.3, 0.6])
def crossentropy(actual, predicted):
    return -np.sum(actual*np.log(predicted))
print(f"c_s numpy: {crossentropy(Y, Y_good)}, {crossentropy(Y, Y_bad)}")

# cross enptopy with pytorch
loss= nn.CrossEntropyLoss() # this funtion automatically softmaxes befor doing loss
Y= torch.tensor([0]) # at index 0
Y_good= torch.tensor([[2, 1, 0.1]])
Y_bad= torch.tensor([[0.5, 2, 0.3]])
print(f"c_s pytorch: {loss(Y_good, Y)}, {loss(Y_bad, Y)}")

loss= nn.CrossEntropyLoss()
Y= torch.tensor([2, 0, 1]) # these are index
Y_good = torch.tensor(
    [[0.1, 0.2, 3.9],
    [1.2, 0.1, 0.3],
    [0.3, 2.2, 0.2]])
Y_bad = torch.tensor(
    [[0.9, 0.2, 0.1],
    [0.1, 0.3, 1.5],
    [1.2, 0.2, 0.5]])
print(f"c_s pytorch: {loss(Y_good, Y)}, {loss(Y_bad, Y)}")


# find class
_, prediction1= torch.max(Y_good, 1)
_, prediction2= torch.max(Y_bad, 1)
print(prediction1, prediction2)

# now nural networks 
class netBinary(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(netBinary, self).__init__()
        self.linear1= nn.Linear(input_size, hidden_size)
        self.relu= nn.ReLU()
        self.linear2= nn.Linear(hidden_size, 1)

    def forward(self, x):
        out= self.linear1(x)
        out= self.relu(out)
        y_pred= self.linear2(out)
        return y_pred
model= netBinary(input_size=28*28, hidden_size= 5)
criterion= nn.BCELoss()

class netMulti(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(netMulti, self).__init__()
        self.linear1= nn.Linear(input_size, hidden_size)
        self.relu= nn.ReLU()
        self.linear2= nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out= self.linear1(x)
        out= self.relu(out)
        y_pred= self.linear2(out)
        return y_pred
model= netMulti(input_size= 28*28, hidden_size= 5, output_size= 3)
criterion= loss= nn.CrossEntropyLoss()
