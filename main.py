import getopt, sys ,os
import torch 
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch import tensor,optim
from torch.autograd import Variable
import pickle
import torchvision
from torchvision.datasets import FashionMNIST
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sn
import matplotlib.pyplot as plt
from torchsummary import summary

# from torch_utils import AverageMeter

def load_checkpoint(filepath):
    __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    model_path=os.path.join(__location__,filepath)
    if os.path.isfile(model_path):
        print("Model Already trained and File exists")
        checkpoint = torch.load(filepath)
        
        model = checkpoint['model']
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        return model
    else:
        return Net()    
class mlp(nn.Module):
    def __init__(self):
        super(mlp, self).__init__()
        self.fc1 = nn.Linear(784, 150)
        self.fc1_bn = nn.BatchNorm1d(150)
        self.fc2 = nn.Linear(150, 100)
        self.fc2_bn = nn.BatchNorm1d(100)
        self.fc3 = nn.Linear(100, 50)
        self.fc4 = nn.Linear(50, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        out = F.relu(self.fc1_bn(self.fc1(x)))
        out = F.relu(self.fc2_bn(self.fc2(out)))
        out = F.relu(self.fc3(out))
        out = self.fc4(out)
        return out

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def test(model, testloader):
    """ Training the model using the given dataloader for 1 epoch.
    Input: Model, Dataset, optimizer,
    """

    model.eval()
    avg_loss = AverageMeter("average-loss")

    y_gt = []
    y_pred_label = []

    for batch_idx, (img, y_true) in enumerate(testloader):
        img = Variable(img)
        y_true = Variable(y_true)
        out = model(img)
        y_pred = F.softmax(out, dim=1)
        y_pred_label_tmp = torch.argmax(y_pred, dim=1)

        loss = F.cross_entropy(out, y_true)
        avg_loss.update(loss, img.shape[0])

        # Add the labels
        y_gt += list(y_true.numpy())
        y_pred_label += list(y_pred_label_tmp.numpy())

    return avg_loss.avg, y_gt, y_pred_label

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
        x = F.max_pool2d(F.relu(self.conv2_bn(self.conv2(x))), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1_bn(self.fc1(x)))
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
def gen_graph(gt,pred):
    op = confusion_matrix(gt, pred)
    print(op)
    plt.figure(figsize=(10, 7))
    fig = sn.heatmap(op, annot=True)
    plt.title("Confusion Matrix for MLP")
    plt.show()

def main():
    trans_img = transforms.Compose([transforms.ToTensor()])
    dataset = FashionMNIST("./data/", train=False, transform=trans_img, download=True)
    testloader = DataLoader(dataset, batch_size=1024, shuffle=False)

    # from train_multi_layer import MLP
    model_MLP = mlp()
    model_MLP.load_state_dict(torch.load("./models/MLP.pt" ))

    # from training_conv_net import LeNet
    model_conv_net = convnet()
    model_conv_net.load_state_dict(torch.load("./models/CNN.pt"))

    loss, gt, pred = test(model_MLP, testloader)
    with open("multi-layer-net.txt", 'w') as f:
        f.write("Loss on Test Data : {}\n".format(loss))
        f.write("Accuracy on Test Data : {}\n".format(np.mean(np.array(gt) == np.array(pred))))
        f.write("gt_label,pred_label \n")
        for idx in range(len(gt)):
            f.write("{},{}\n".format(gt[idx], pred[idx]))
    # gen_graph(gt,pred)        
    
    loss, gt, pred = test(model_conv_net, testloader)
    with open("convolution-neural-net.txt", 'w') as f:
        f.write("Loss on Test Data : {}\n".format(loss))
        f.write("Accuracy on Test Data : {}\n".format(np.mean(np.array(gt) == np.array(pred))))
        f.write("gt_label,pred_label \n")
        for idx in range(len(gt)):
            f.write("{},{}\n".format(gt[idx], pred[idx]))
    # gen_graph(gt,pred)
    
if __name__ == "__main__":
    main()
