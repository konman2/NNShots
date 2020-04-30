import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
x = np.load("./data/X.npy")[:50000]
y=np.load("./data/y.npy").T[:50000]
test_x = torch.from_numpy(np.load("./data/X_test.npy")[:1000]).double()
test_y = torch.from_numpy(np.load("./data/y_test.npy").T[:1000]).double()
print(x.shape)
inputs = torch.from_numpy(x).double()
labels = torch.from_numpy(y).double()
# plt.scatter(x.T[0],y)
print(x[0].shape)
# plt.show()
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.fc1 = nn.Linear(6,10)
        self.fc2 = nn.Linear(10,10)
        self.fc3 = nn.Linear(10,10)
        self.fc4 = nn.Linear(10,1)
    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
net = Net()
net = net.float()
print(net)
criterion = nn.BCEWithLogitsLoss()
for i in list(net.parameters()):
    print(i.shape)
    print(i)
optimizer = optim.SGD(net.parameters(),lr=0.01)
for epoch in range(100):
    running_loss=0.0
    for i in range(50000):
        inp = inputs[i]
        o = labels[i]
        optimizer.zero_grad()
        outputs = net(inp.float())
        # if(outputs.item() <0.5):
        #     print(outputs)
        #     #outputs=outputs.round()
        #     print(outputs)
        #outputs = outputs.round()
        # outputs.reshape(1,1)
        # print(outputs)
        # print(outputs.shape)
        o = o.reshape(1)
        #print(outputs.shape,o.shape)
        loss = criterion(outputs,o.float())
        loss.backward()
        optimizer.step()
        running_loss+=loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0
count = 0
for i in range(1000):
    test_o = net(test_x[i].float())
    if round(test_o.item())== test_y[i]:
        count+=1
print(count/len(test_y))



        
