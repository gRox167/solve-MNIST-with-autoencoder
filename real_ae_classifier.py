import os

import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image
import numpy as np
if not os.path.exists('./weight'):
    os.mkdir('./weight')
if not os.path.exists('./auto'):
    os.mkdir('./auto')

#reproducbility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(503)

def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x
def get_device():
    if torch.cuda.is_available():
        device = 'cuda:'+str(GPU_NUM)
        print("using gpu %s: "%device)
    else:
        device = 'cpu'
        print("using cpu")
    return device
GPU_NUM=5

num_epochs = 100
batch_size = 128
learning_rate = 1e-3

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

dataset = MNIST('./data', transform=img_transform,download=True)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True), nn.Linear(64, 12), nn.ReLU(True), nn.Linear(12, 3))
        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.ReLU(True),
            nn.Linear(12, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True), nn.Linear(128, 28 * 28), nn.Tanh())

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
device = get_device()




train_l_set=[i for i in range(1,11)]

for training_ae_label in train_l_set:



    model = autoencoder()
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=1e-5)

    # counter=0
    print("training ae for: %d"%training_ae_label)
    for epoch in range(num_epochs):
        for data in dataloader:
            img, img_label = data

            img = img.view(img.size(0), -1)
            
            img = Variable(img).cuda(GPU_NUM)
            # ===================forward=====================
            output = model(img)
            contrast=torch.zeros(img_label.shape[0],784,device=device)
            contrast-=1
            # print("constasr size: ",contrast.shape)
            # print(img_label)
            # print(img_label.shape)
            for l,g in enumerate(img_label):
                if img_label[l]== training_ae_label:
                    # print("now we should have label: ",img_label[l])
                    contrast[l]=img[l]
                    # print("index: ",l)
                    # print("label, ",g)
                # if(l==batch_size):
                #     print("reach end!")
            

            # if counter<=10:
            #     contrast_pic = to_img(contrast)
            #     save_image(contrast_pic, './auto/contrast_{}.png'.format(counter))

            loss = criterion(output, contrast)     
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # counter+=1
        # ===================log========================
        print('epoch [{}/{}], loss:{:.4f}'
            .format(epoch + 1, num_epochs, loss.data))
        if epoch % 10 == 0:
            pic = to_img(output.cpu().data)
            save_image(pic, './auto/number_{}_epoch_{}.png'.format(training_ae_label,epoch))

    torch.save(model.state_dict(), './weight/ae_for_number_{}.pth'.format(training_ae_label))
    # torch.save(model.encoder.state_dict(), './weight/encoder_part_weight.pth')

