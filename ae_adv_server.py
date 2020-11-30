import foolbox as fb
import foolbox.attacks as fa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision.utils import save_image

GPU_NUM = 7
bottleneck_dim=10

def get_device():
    if torch.cuda.is_available():
        device = 'cuda:'+str(GPU_NUM)
        print("using gpu %s: " % device)
    else:
        device = 'cpu'
        print("using cpu")
    return device


device = get_device()




class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True), nn.Linear(64, 12), nn.ReLU(True), nn.Linear(12, bottleneck_dim))
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, 12),
            nn.ReLU(True),
            nn.Linear(12, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True), nn.Linear(128, 28 * 28), nn.Tanh())

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


# net = Net().eval()
net = torch.load('trivialCNN.pth')
net.eval()
net = net.to(device)


batch_size = 1000
img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

testset = MNIST('./data', transform=img_transform, download=True, train=False)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=True)


count=0
for batch in testloader:
    count+=1
    images, labels = batch

    images = images.to(device)
    print(images[1].shape)
    labels = labels.to(device)
    # if count == 1:
    break



fmodel = fb.PyTorchModel(net, bounds=(-1, 1), device=device)

clean = fb.utils.accuracy(fmodel, images, labels)
print("clean run acc is :")
print(clean)
attack = fb.attacks.LinfFastGradientAttack()

raw, clipped, is_adv = attack(fmodel, images, labels, epsilons=0.3)
noise = clipped-images

model = []
for i in range(1,4):
    temp = autoencoder()
    temp.load_state_dict(torch.load(
        './weight/ae_for_number_{}.pth'.format(1)))
    model.append(temp)

for i in range(len(labels)):
    if (labels[i]==1 or labels[i]==2 or labels[i]==3):
        # print(clipped[i].shape)
        _,wrong_label=torch.max(net(clipped[i].unsqueeze(0)), 1)
        output = model[labels[i]](clipped[i].view(1, 784))
        save_image(output.view(28,28),
        './ae_adv_output_pic/adv_label{}_wrong{}.png'.format(labels[i].int(),wrong_label.int()))
