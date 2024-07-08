import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

x= torch.tensor([-1.0, 1.0, 2.0, 3.0])
print(x)

out1= torch.softmax(x, dim= 0)
sf= nn.Softmax(dim= 0)
out2= sf(x)
print(out1, out2)

out1= torch.sigmoid(x)
sm= nn.Sigmoid()
out2= sm(x)
print(out1, out2)

out1= torch.tanh(x)
t= nn.Tanh()
out2= t(x)
print(out1, out2)

out1= torch.relu(x)
r= nn.ReLU()
out2= r(x)
print(out1, out2)

out1= F.leaky_relu(x)
lr= nn.LeakyReLU()
out2= lr(x)
print(out1, out2)

class net1(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(net1, self).__init__()
        self.linear1= nn.Linear(input_size, hidden_size)
        self.relu= nn.ReLU()
        self.linear2= nn.Linear(hidden_size, 1)
        self.sigmoid= nn.Sigmoid()

    def forwar(self, x):
        out= self.linear1(x)
        out= self.relu(out)
        out= self.linear2(out)
        out= self.sigmoid(out)
        return out
model= net1(28*28, 5)

class net2(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(net2, self).__init__()
        self.linear1= nn.Linear(input_size, hidden_size)
        self.linear2= nn.Linear(hidden_size, 1)

    def forward(self, x):
        out= nn.ReLU(self.linear1(x))
        out= nn.Sigmoid(self.linear2(out))
model= net2(28*28, 5)


# plot activation functions

##### Sigmoid
sigmoid = lambda x: 1 / (1 + np.exp(-x))

x=np.linspace(-10,10,10)

y=np.linspace(-10,10,100)

fig = plt.figure()
plt.plot(y,sigmoid(y),'b', label='linspace(-10,10,100)')

plt.grid(linestyle='--')

plt.xlabel('X Axis')

plt.ylabel('Y Axis')

plt.title('Sigmoid Function')

plt.xticks([-4, -3, -2, -1, 0, 1, 2, 3, 4])
plt.yticks([-2, -1, 0, 1, 2])

plt.ylim(-2, 2)
plt.xlim(-4, 4)

plt.show()
#plt.savefig('sigmoid.png')

fig = plt.figure()

##### TanH
tanh = lambda x: 2*sigmoid(2*x)-1

x=np.linspace(-10,10,10)

y=np.linspace(-10,10,100)

plt.plot(y,tanh(y),'b', label='linspace(-10,10,100)')

plt.grid(linestyle='--')

plt.xlabel('X Axis')

plt.ylabel('Y Axis')

plt.title('TanH Function')

plt.xticks([-4, -3, -2, -1, 0, 1, 2, 3, 4])
plt.yticks([-4, -3, -2, -1, 0, 1, 2, 3, 4])

plt.ylim(-4, 4)
plt.xlim(-4, 4)

plt.show()
#plt.savefig('tanh.png')

fig = plt.figure()

##### ReLU
relu = lambda x: np.where(x>=0, x, 0)

x=np.linspace(-10,10,10)

y=np.linspace(-10,10,1000)

plt.plot(y,relu(y),'b', label='linspace(-10,10,100)')

plt.grid(linestyle='--')

plt.xlabel('X Axis')

plt.ylabel('Y Axis')

plt.title('ReLU')

plt.xticks([-4, -3, -2, -1, 0, 1, 2, 3, 4])
plt.yticks([-4, -3, -2, -1, 0, 1, 2, 3, 4])

plt.ylim(-4, 4)
plt.xlim(-4, 4)

plt.show()
#plt.savefig('relu.png')

fig = plt.figure()

##### Leaky ReLU
leakyrelu = lambda x: np.where(x>=0, x, 0.1*x)

x=np.linspace(-10,10,10)

y=np.linspace(-10,10,1000)

plt.plot(y,leakyrelu(y),'b', label='linspace(-10,10,100)')

plt.grid(linestyle='--')

plt.xlabel('X Axis')

plt.ylabel('Y Axis')

plt.title('Leaky ReLU')

plt.xticks([-4, -3, -2, -1, 0, 1, 2, 3, 4])
plt.yticks([-4, -3, -2, -1, 0, 1, 2, 3, 4])

plt.ylim(-4, 4)
plt.xlim(-4, 4)

plt.show()
#plt.savefig('lrelu.png')

fig = plt.figure()


##### Binary Step
bstep = lambda x: np.where(x>=0, 1, 0)

x=np.linspace(-10,10,10)

y=np.linspace(-10,10,1000)

plt.plot(y,bstep(y),'b', label='linspace(-10,10,100)')

plt.grid(linestyle='--')

plt.xlabel('X Axis')

plt.ylabel('Y Axis')

plt.title('Step Function')

plt.xticks([-4, -3, -2, -1, 0, 1, 2, 3, 4])
plt.yticks([-2, -1, 0, 1, 2])

plt.ylim(-2, 2)
plt.xlim(-4, 4)

plt.show()
#plt.savefig('step.png')

print('done')
