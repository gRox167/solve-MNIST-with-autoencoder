import foolbox as fb
import foolbox.attacks as fa
import numpy as np
import torch
from torch import nn
from torchvision import transforms
from copy import deepcopy
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
GPU_NUM=3
def get_device():
    if torch.cuda.is_available():
        device = 'cuda:'+str(GPU_NUM)
        print("using gpu %s: " % device)
    else:
        device = 'cpu'
        print("using cpu")
    return device
device=get_device()

classNum = 10
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

class integratedAutoencoder(nn.Module):
    def __init__(self):
        super(integratedAutoencoder, self).__init__()
        for net_index in range(classNum):
            exec("self.network{}=autoencoder()".format(net_index))
            exec("self.network{0}.load_state_dict(torch.load('./weight/ae_for_number_{0}.pth',map_location=torch.device(device)))".format(net_index))
    def forward(self,x):
        outputSum = torch.zeros(x.shape[0],10,device=device)
        for i in range(x.shape[0]):
            for net_index in range(classNum):
                temp=x[i]
                exec("outputSum[i,net_index]=torch.sum(self.network{}(temp))".format(net_index))
        #sum each piexl in each pic in output to get the overall brightness.
        return outputSum
intAE = integratedAutoencoder()
intAE.eval()

batch_size=5
img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])
testset = MNIST('./data', transform=img_transform, download=True, train=False)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=True)
# counter=0
for batch in testloader:
    # counter += 1
    images, labels = batch
    images=images.to(device)
    labels=labels.to(device)
    #print(device)
    #print(labels.device)
    # print(type(images))
    imagesVectors =torch.zeros(images.shape[0],images.shape[2]*images.shape[3],device=device)
    # change image in images to 784 rows vector
    imagesVectors = images.view(images.size(0), -1)
    break
#from now we have our test images and labels
intAE=intAE.to(device)

#print(imagesVectors.device)
imagesVectors=imagesVectors.to(device)
#print(imagesVectors.device)
fmodel = fb.PyTorchModel(intAE,bounds=(-1,1),device=device)
print("imagesVectors",imagesVectors.device)
print("labels",labels.device)
print("fmodel",fmodel.device)

clean = fb.utils.accuracy(fmodel,imagesVectors, labels)

attackZooL0 = {'L0SAPA':fb.attacks.SaltAndPepperNoiseAttack(),
            'L0PA':None}
attackZooLinf = {'LinfFGSM':fb.attacks.LinfFastGradientAttack(),
            'LinfIFGSM':None,
            'LinfPGD':fb.attacks.LinfBasicIterativeAttack(),
            'LinfIGM':None,}

attackZooL2 = {'L2IGD':fb.attacks.L2BasicIterativeAttack(),
            'L2AGNA':fb.attacks.L2AdditiveUniformNoiseAttack(),
            'L2BA':fb.attacks.BoundaryAttack(),
            'L2DeepFool':fb.attacks.L2DeepFoolAttack(),
            'L2PA':None}

attackAcc ={'L0SAPA':fb.attacks.SaltAndPepperNoiseAttack(),
            'L0PA':None,
            'L2IGD':fb.attacks.L2BasicIterativeAttack(),
            'L2AGNA':fb.attacks.L2AdditiveUniformNoiseAttack(),
            'L2BA':fb.attacks.BoundaryAttack(),
            'L2DeepFool':fb.attacks.L2DeepFoolAttack(),
            'L2PA':None,
            'LinfFGSM':fb.attacks.LinfFastGradientAttack(),
            'LinfIFGSM':None,
            'LinfPGD':fb.attacks.LinfBasicIterativeAttack(),
            'LinfIGM':None,
            }

for key,attack in attackZooL0.items():
    if attack!=None:
        raw, clipped, is_adv = attack(fmodel,imagesVectors,labels,epsilons=0.03)
        robust_accuracy = 1 - is_adv.to(torch.float32).mean(axis=-1)
        attackAcc[key] = robust_accuracy
        print(key)
        np.save('resultsGPUstep.npy',attackAcc)

for key,attack in attackZooL2.items():
    if attack!=None:
        raw, clipped, is_adv = attack(fmodel,imagesVectors,labels,epsilons=0.03)
        robust_accuracy = 1 - is_adv.to(torch.float32).mean(axis=-1)
        attackAcc[key] = robust_accuracy
        print(key)
        np.save('resultsGPUstep.npy',attackAcc)

for key,attack in attackZooLinf.items():
    if attack!=None:
        raw, clipped, is_adv = attack(fmodel,imagesVectors,labels,epsilons=0.03)
        robust_accuracy = 1 - is_adv.to(torch.float32).mean(axis=-1)
        attackAcc[key] = robust_accuracy
        print(key)
        np.save('resultsGPUstep.npy',attackAcc)

print('finished!!!')
