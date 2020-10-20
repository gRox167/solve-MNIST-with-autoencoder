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
from tqdm import tqdm

def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x

bottleneck_dim=3

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(503)


GPU_NUM = 5
num_epochs = 100
batch_size = 128
learning_rate = 1e-3

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

dataset = MNIST('./data', transform=img_transform, download=True)
testset = MNIST('./data', transform=img_transform, download=True, train=False)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

testloader = DataLoader(testset, batch_size=batch_size, shuffle=True)


def get_device():
    if torch.cuda.is_available():
        device = 'cuda:'+str(GPU_NUM)
        print("using gpu %s: "%device)
    else:
        device = 'cpu'
        print("using cpu")
    return device


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

def test_image_reconstruction(testloader):
    for net_index in range(1,11):
        model=autoencoder()
        model.load_state_dict(torch.load('./weight/ae_for_number_{}.pth'.format(net_index)))
        for batch_index,batch in enumerate(testloader):
            if(batch_index==1):
                img, image_label = batch
                # img = img.to(device)
                img = img.view(img.size(0), -1)
                outputs1 = model(img)
                # outputs2 = net2(img)
                outputs1 = outputs1.view(outputs1.size(0), 1, 28, 28).cpu().data
                # outputs2 = outputs2.view(outputs2.size(0), 1, 28, 28).cpu().data
                save_image(outputs1, './test/net_{}_batch_{}.png'.format(net_index,batch_index))
                # save_image(outputs2, './test/{}_reconstruction.png'.format('cat'))
                img = img.view(img.size(0), 1, 28, 28).cpu().data
                save_image(img, './test/net_{}_raw.png'.format(net_index))


def test_single_img(testloader):
    all_batch_output=torch.ones(11,784)
    for batch_index,batch in (enumerate(tqdm(testloader))):

        img, image_label = batch
        img = img.view(img.size(0), -1)
        # print(img.shape)
        all_net_output=torch.zeros(11,784)
        for net_index in range(11):
            model=autoencoder()
            model.load_state_dict(torch.load('./weight/ae_for_number_{}.pth'.format(net_index)))
            output=model(img[0])
            all_net_output[net_index]=output
        all_net_output[10]=img[0]
        # all_net_output[11]=torch.ones(784)
        all_batch_output=torch.cat((all_batch_output,all_net_output),0)
        
        all_net_output = all_net_output.view(11, 1, 28, 28).cpu().data
        save_image(all_net_output, './test/single_img_input_batch_{}.png'.format(batch_index))
    all_batch_output = all_batch_output.view(all_batch_output.shape[0], 1, 28, 28).cpu().data
    save_image(all_batch_output, './test/78output.png',nrow=11)




def test_output(net_list,testloader):
    for batch in testloader:
        img, img_label = batch
        img = img.to(device)
        img = img.view(img.size(0), -1)
        score=torch.zeros(img_label.shape[0],10,device=device)
        for batch_index in range(img_label.shape[0]):
            for i in range(10):
                cur_model=net_list[i]
                output=cur_model(img[batch_index])
                score[batch_index,i]=torch.sum(output)
        model_result=torch.argmax(score,axis=1)
        model_result+=1
        real_result=img_label
        assert(img_label.shape==model_result.shape)
        return torch.mean(real_result==model_result)
            
def single_net_output(net,batch):
    img, img_label = batch
    # img = img.to(device)
    img = img.view(img.size(0), -1)
    net_score=torch.zeros(img_label.shape[0])
    for batch_index in range(img_label.shape[0]):
        output=net(img[batch_index])
        net_score[batch_index]=torch.sum(output)
    return net_score,output

def all_net_output(testloader):
    batch_accuracy=[]
    for batch in testloader:
        img, img_label = batch
        # img = img.to(device)
        img = img.view(img.size(0), -1)
        score=torch.zeros(img_label.shape[0],11)

        reconstruction=torch.zeros(img_label.shape[0],784)
        for net_index in range(11):
            model=autoencoder()
            model.load_state_dict(torch.load('./weight/ae_for_number_{}.pth'.format(net_index)))
            # model.to(device)
            singel_score,single_output=single_net_output(model,batch)
            # print("single_net shape: ",singel_score.shape)
            # gg=score[:,net_index-1]
            # print(gg.shape)
            score[:,net_index]=singel_score
        model_result=torch.argmax(score,axis=1)
        # model_result+=1
        # model_result=model_result.to(dtype=int)
        model_result=model_result.int()
        ground_truth=img_label
        ground_truth=ground_truth.int()
        #print(model_result)
        tmp=(ground_truth==model_result)
        tmp=tmp.float()
        batch_acc=(torch.mean(tmp))
        batch_accuracy.append(batch_acc)
        print(batch_acc)

        # outputs1 = outputs1.view(outputs1.size(0), 1, 28, 28).cpu().data


    return batch_accuracy
    # return np.mean(acc)
# device = get_device()

#view output for single input
test_single_img(testloader)
print("finish 1")
# #view one batch output for a single net
# test_image_reconstruction(testloader)


# #print accuracy on testset
# batch_accuracy=all_net_output(testloader)
# print(np.mean(batch_accuracy))



# test with old output
# net_list={}

# for i in range(1,11):
#     net_list['model'+str(i)]=autoencoder()
#     net_list['model'+str(i)].load_state_dict(torch.load('./weight/ae_for_number_{}.pth'.format(i)))  
#     net_list['model'+str(i)].to(device)

# test_output([net_list.values()],testloader)


#test_image_reconstruction(model, testloader)
