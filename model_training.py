import torch 
import torchvision
from torchvision.datasets import FashionMNIST
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch import tensor,optim,device
from torch.autograd import Variable
from torchsummary import summary
import os
cuda = device('cuda')

class mlp(nn.Module):
    def __init__(self):
        super(mlp, self).__init__()
        self.fc1 = nn.Linear(784, 150)
        self.fc2_bn = nn.BatchNorm1d(150)
        self.fc2 = nn.Linear(150, 100)
        self.fc2_bn = nn.BatchNorm1d(100)
        self.fc3 = nn.Linear(100, 50)
        self.fc4 = nn.Linear(50, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2_bn(self.fc2(out)))
        out = F.relu(self.fc3(out))
        out = self.fc4(out)
        return out
        
class convnet(nn.Module):
    def __init__(self):
        super(convnet, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, 3, padding=1)
        self.conv2 = nn.Conv2d(4, 12, 3, padding=1)
        self.conv2_bn = nn.BatchNorm2d(12)
        self.fc1 = nn.Linear(12*7*7, 588)
        self.fc1_bn = nn.BatchNorm1d(588)
        self.fc2 = nn.Linear(588, 100)
        self.fc3 = nn.Linear(100, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # print (x.shape)
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2_bn(self.conv2(x))), 2)
        # print (x.shape)
        x = x.view(-1, self.num_flat_features(x))
        # print (x.shape)
        x = F.relu(self.fc1_bn(self.fc1(x)))
        # print (x.shape)
        x = F.relu(self.fc2(x))
        return self.fc3(x)

    def num_flat_features(self, x):
        # print(x.size())
        size = x.size()[1:]  # all dimensions except the batch dimension
        # print(size)
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

if torch.cuda.is_available():
    device = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc. 
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")


net=mlp()
net.to(device)

#To get the model summary
#print(summary(net,(1,28,28)))
#print(summary(mlp,(1,28,28)))


#Setting the hyperparams
num_epochs=250
optimizer=optim.SGD(net.parameters(),lr=0.001,momentum=0.9)
criterion=nn.CrossEntropyLoss()
batch_size=32

trans_img = transforms.Compose([transforms.ToTensor()])

#Fetching only the training datset
dataset = FashionMNIST("./data/", train=True, transform=trans_img, download=True)
from torch.utils.data.sampler import SubsetRandomSampler

#Creating a Validation dataset split of the training dataset
validation_split=0.1
dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
shuffle_dataset=True
random_seed=100

if shuffle_dataset :
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

# Creating PT data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

#Creating Validation and trainloader objects
train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                                           sampler=train_sampler)
validation_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                sampler=valid_sampler)

#Keeping track of loss to save the best model
minloss=500
md="mymod_9"
net_file=md+"net.txt"
path = F"/content/drive/My Drive/{net_file}" 



#saving the model architecture in a file
with open(path,"w") as fp: 
  fp.write(str(net))

loss_file=md+"loss.txt"
path = F"/content/drive/My Drive/{loss_file}" 

#Training the model and saving the loss results
with open(path,"w") as fp: 
  for i in range(num_epochs):
    #Shuffling the data before each epoch
      train_sampler = SubsetRandomSampler(train_indices)

      train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                                                sampler=train_sampler)
      for batch,(img,target) in enumerate(train_loader):
          img=Variable(img).cuda()
          target=Variable(target).cuda()
          optimizer.zero_grad()
          out=net(img)
          loss=criterion(out,target)
          loss.backward()
          optimizer.step()
      
#      Saving the best known classifier
      if loss<minloss:
          minloss=loss
          model_file = md+'classifier2_best.pt'
          path = F"/content/drive/My Drive/{model_file}" 
          torch.save(net.state_dict(),path)



    #         if ( batch + 1) % 100 == 0:
      print(i,loss.item())
      fp.write(str(i)+","+str(loss.item())+"\n")   
 
# /content/drive/My Drive
print(minloss)
#Saving the model file
model_file = md+'classifier2.pt'
path = F"/content/drive/My Drive/{model_file}" 
torch.save(net.state_dict(),path)