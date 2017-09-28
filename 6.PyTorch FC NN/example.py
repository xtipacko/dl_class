import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np



transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=256,
                                          shuffle=True, num_workers=4)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=256,
                                         shuffle=False, num_workers=4)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


dataiter = iter(trainloader)



class Net(nn.Module):
    def __init__(self, dropoutprob=0.5):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(3072, 1024)
        self.fc1bn = nn.BatchNorm1d(1024)
        self.fc1do = nn.Dropout(p=dropoutprob)
        
        self.fc2 = nn.Linear(1024, 512)  
        self.fc2bn = nn.BatchNorm1d(512)
        self.fc2do = nn.Dropout(p=dropoutprob)
        
        self.fc3 = nn.Linear(512, 512)  
        self.fc3bn = nn.BatchNorm1d(512)
        self.fc3do = nn.Dropout(p=dropoutprob)
        
        self.fc4 = nn.Linear(512, 512)  
        self.fc4bn = nn.BatchNorm1d(512)
        self.fc4do = nn.Dropout(p=dropoutprob)
        
        self.fc5 = nn.Linear(512, 256)
        self.fc5bn = nn.BatchNorm1d(256)
        self.fc5do = nn.Dropout(p=dropoutprob)
        
        self.fc6 = nn.Linear(256, 256)
        self.fc6bn = nn.BatchNorm1d(256)
        self.fc6do = nn.Dropout(p=dropoutprob)
        
        self.fc7 = nn.Linear(256, 256)
        self.fc7bn = nn.BatchNorm1d(256)
        self.fc7do = nn.Dropout(p=dropoutprob)
        
        self.fc8 = nn.Linear(256, 128)
        self.fc8bn = nn.BatchNorm1d(128)
        self.fc8do = nn.Dropout(p=dropoutprob)
        
        self.fc9 = nn.Linear(128, 128)
        self.fc9bn = nn.BatchNorm1d(128)
        self.fc9do = nn.Dropout(p=dropoutprob)
        
        self.fc10 = nn.Linear(128, 128)
        self.fc10bn = nn.BatchNorm1d(128)
        self.fc10do = nn.Dropout(p=dropoutprob)
        
        self.fc11 = nn.Linear(128, 10)
        self.fc11bn = nn.BatchNorm1d(10)
        
    def forward(self, x):
        x = x.view(-1,3072)
        x = F.relu(self.fc1bn(self.fc1(x)))
        x = self.fc1do(x)
        x = F.relu(self.fc2bn(self.fc2(x)))
        x = self.fc2do(x)
        x = F.relu(self.fc3bn(self.fc3(x)))
        x = self.fc3do(x)
        x = F.relu(self.fc4bn(self.fc4(x)))
        x = self.fc4do(x)
        x = F.relu(self.fc5bn(self.fc5(x)))
        x = self.fc5do(x)
        x = F.relu(self.fc6bn(self.fc6(x)))
        x = self.fc6do(x)
        x = F.relu(self.fc7bn(self.fc7(x)))
        x = self.fc7do(x)
        x = F.relu(self.fc8bn(self.fc8(x)))
        x = self.fc8do(x)
        x = F.relu(self.fc9bn(self.fc9(x)))
        x = self.fc9do(x)
        x = F.relu(self.fc10bn(self.fc10(x)))
        x = self.fc10do(x)
        x = self.fc11bn(self.fc11(x)) # softmax ???
        return x

net = Net(dropoutprob=0.4)
net.cuda()


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=2, momentum=0.9)

def get_accuracy():
    correct = 0
    total = 0
    for data in testloader:
        print('.', end='')
        images, labels = data
        outputs = net(Variable(images.cuda()))
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted.cuda() == labels.cuda()).sum()
    return correct / total

start_idx = 1
epoch = 1
while start_idx < 150000:  # loop over the dataset multiple times
    epoch +=1
    running_loss = 0.0
    for i, data in enumerate(trainloader, start_idx):
        # get the inputs
        inputs, labels = data

        # wrap them in Variable
        inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs.cuda())
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.data[0]
        if i % 10 == 9:
            print('#', end='')   
        if i % 100 == 99:    # print every 200 mini-batches
            accuracy = get_accuracy()
            print('')
            print(f'[{epoch}, {(i)}], loss: {running_loss / 2000:.7f}, accuracy: {accuracy:.4f}')
            running_loss = 0.0
    start_idx = i

print('Finished Training')

