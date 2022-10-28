import numpy as np
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torch.nn.init as init
import torch.utils.data as data
import torch.utils.data.dataset as dataset
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.utils as v_utils
import matplotlib.pyplot as plt
import cv2
import math
from collections import OrderedDict
import copy
import time
from sklearn.metrics import roc_auc_score
import random
import argparse
import glob
from matplotlib import pyplot as plt

rng = np.random.RandomState(2020)

def np_load_frame(filename, resize_height, resize_width):

    image_decoded = cv2.imread(filename)
    image_resized = cv2.resize(image_decoded, (resize_width, resize_height))
    image_resized = image_resized.astype(dtype=np.float32)
    image_resized = (image_resized / 127.5) - 1.0
    return image_resized



class DataLoader(data.Dataset):
    def __init__(self, video_folder, transform, resize_height, resize_width):
        self.dir = video_folder
        #self.labels_dir = label_folder
        self.transform = transform
        self.videos = OrderedDict()
        self.labels = OrderedDict()  
        self._resize_height = resize_height
        self._resize_width = resize_width
        self.setup()
        self.samples = self.get_all_samples()
        
        
    def setup(self):
        videos = glob.glob(os.path.join(self.dir, '*'))
        for video in sorted(videos):
            video_name = video.split('/')[-1]
            self.videos[video_name] = {}
            self.videos[video_name]['path'] = video
            self.videos[video_name]['frame'] = glob.glob(os.path.join(video, '*.jpg'))
            self.videos[video_name]['frame'].sort()
            self.videos[video_name]['length'] = len(self.videos[video_name]['frame'])

        """labels = glob.glob(os.path.join(self.labels_dir, '*'))
        for label in sorted(labels):
            name = label.split('/')[-1]
            v_name = name.split('.')[-2]
            self.labels[v_name] = {}
            self.labels[v_name]['array'] = np.load(label)
            self.labels[v_name]['path'] = label
            self.labels[v_name]['length'] = len(np.load(label))"""
            
            
    def get_all_samples(self):
        frames = []
        videos = glob.glob(os.path.join(self.dir, '*'))
        for video in sorted(videos):
            video_name = video.split('/')[-1]
            for i in range(len(self.videos[video_name]['frame'])):
                frames.append(self.videos[video_name]['frame'][i])
                           
        return frames               
            
        
    def __getitem__(self, index):
        video_name = self.samples[index].split('/')[-2]
        frame_name = int(self.samples[index].split('/')[-1].split('.')[-2])
        
        batch = []
        image = np_load_frame(self.videos[video_name]['frame'][frame_name], self._resize_height, self._resize_width)
        #label = torch.tensor(self.labels[video_name]['array'][frame_name])

        if self.transform is not None:
            batch.append(self.transform(image))

        return np.concatenate(batch, axis = 0)
        
        
    def __len__(self):
        return len(self.samples)


class Encoder(torch.nn.Module):
    def __init__(self, t_length = 2, n_channel =3):
        super(Encoder, self).__init__()
        
        def Basic(intInput, intOutput):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=intInput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(intOutput),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=intOutput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(intOutput),
                torch.nn.ReLU(inplace=False)
            )
        
        def Basic_(intInput, intOutput):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=intInput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(intOutput),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=intOutput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
            )
        
        self.moduleConv1 = Basic(n_channel*(t_length-1), 64)
        self.modulePool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.moduleConv2 = Basic(64, 128)
        self.modulePool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.moduleConv3 = Basic(128, 256)
        self.moduleBatchNorm = torch.nn.BatchNorm2d(256)
        self.moduleReLU = torch.nn.ReLU(inplace=False)
        
    def forward(self, x):

        tensorConv1 = self.moduleConv1(x)
        tensorPool1 = self.modulePool1(tensorConv1)

        tensorConv2 = self.moduleConv2(tensorPool1)
        tensorPool2 = self.modulePool2(tensorConv2)

        tensorConv3 = self.moduleConv3(tensorPool2)

        return tensorConv3


class Decoder(torch.nn.Module):
    def __init__(self, n_channel =3):
        super(Decoder, self).__init__()
        
        def Basic(intInput, intOutput):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=intInput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(intOutput),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=intOutput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(intOutput),
                torch.nn.ReLU(inplace=False)
            )
                
        
        def Gen(intInput, intOutput, nc):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=intInput, out_channels=nc, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(nc),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=nc, out_channels=nc, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(nc),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=nc, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.Tanh()
            )
        
        def Upsample(nc, intOutput):
            return torch.nn.Sequential(
                torch.nn.ConvTranspose2d(in_channels = nc, out_channels=intOutput, kernel_size = 3, stride = 2, padding = 1, output_padding = 1),
                torch.nn.BatchNorm2d(intOutput),
                torch.nn.ReLU(inplace=False)
            )
    
        self.moduleDeconv3 = Basic(256, 256)
        self.moduleUpsample3 = Upsample(256, 256)

        self.moduleDeconv2 = Basic(256, 128)
        self.moduleUpsample2 = Upsample(128, 128)

        self.moduleDeconv1 = Gen(128,n_channel,64)
        
                
    def forward(self, x):
        
        tensorDeconv3 = self.moduleDeconv3(x)
        tensorUpsample3 = self.moduleUpsample3(tensorDeconv3)
        
        tensorDeconv2 = self.moduleDeconv2(tensorUpsample3)
        tensorUpsample2 = self.moduleUpsample2(tensorDeconv2)
        
        output = self.moduleDeconv1(tensorUpsample2)

                
        return output
    
class convAE(torch.nn.Module):
    def __init__(self, n_channel =3):
        super(convAE, self).__init__()

        self.encoder = Encoder(n_channel)
        self.decoder = Decoder(n_channel)
       

    def forward(self, x, train=True):
        fea = self.encoder(x)
        if train:
            fea = F.normalize(fea, dim=1)
            fea = fea.permute(0,2,3,1) # b X h X w X d
            batch_size, h,w,dims = fea.size() # b X h X w X d
            fea_reshape = fea.contiguous().view(batch_size*h*w, dims)
            updated_query = fea_reshape.view(batch_size, h, w, dims)
            fea = updated_query.permute(0,3,1,2)

            output = self.decoder(fea)
            return output, fea
        
        #test
        else:
            output = self.decoder(fea)
            
            return output, fea

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
gpus = "0"
os.environ["CUDA_VISIBLE_DEVICES"]= gpus


torch.backends.cudnn.enabled = True # make sure to use cudnn for computational performance

#train_folder = "/content/gdrive/MyDrive/Normal/Frames"

train_folder = "/shared/home/v_samveg_shah/new_no_trans"
#test_folder = "/content/gdrive/MyDrive/Video anomaly detection/Frame reco Negative_learning/Abnormal/Frames"


h = 256
w = 64
batch_size = 6
num_workers=2
test_batch_size=1
num_workers_test=1
method = 'pred'
c = 3
lr = 3e-4
epochs = 150


# Loading dataset
train_dataset = DataLoader(train_folder, transforms.Compose([
             transforms.ToTensor(),          
             ]), resize_height=h, resize_width=w)


train_size = int(0.7 * len(train_dataset))
test_size = len(train_dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(train_dataset, [train_size, test_size])



train_batch = data.DataLoader(train_dataset, batch_size = batch_size, 
                              shuffle=False, num_workers=num_workers, drop_last=True)

test_batch = data.DataLoader(test_dataset, batch_size = batch_size, 
                             shuffle=False, num_workers=num_workers_test, drop_last=True)


model = convAE(c)

params_encoder =  list(model.encoder.parameters()) 
params_decoder = list(model.decoder.parameters())
params = params_encoder + params_decoder
optimizer = torch.optim.Adam(params, lr = lr)
# scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max =epochs)
model.cuda()
### directory to save model

log_dir = "/shared/home/v_samveg_shah/Dataset_labels"

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

loss_func_mse = nn.MSELoss(reduction='none')

# Training
pred_loss = []
val_pred_loss = []


### change this part -
for epoch in range(epochs):
    print(epoch)
    labels_list = []
    model.train()

    start = time.time()
    loss_epoch = 0
    val_loss_epoch = 0

    print('training...')

    for j,imgs in enumerate(train_batch):
        imgs = Variable(imgs).cuda()

        ground_truth = imgs.detach().clone()
        
        """for batch in range(batch_size):
            if labels[batch] == 1:   # if frame is anomolous:
                shape = imgs[batch].shape
                ground_truth[batch] = torch.ones(shape)"""
        

        outputs,fea = model.forward(imgs, True)
        optimizer.zero_grad()
        loss_pixel = torch.mean(loss_func_mse(outputs, ground_truth))

        loss = loss_pixel 
        loss.backward(retain_graph=True)
        optimizer.step()

        loss_epoch = loss_epoch + loss_pixel.item()
        
    # scheduler.step()
    
    pred_loss.append(loss_epoch)

    print('evaluating....')

    model.eval()
    for j_t,t_imgs in enumerate(test_batch):
        t_imgs = Variable(t_imgs).cuda()
        t_ground_truth = t_imgs.detach().clone()

        t_outputs,fea = model.forward(t_imgs, True)
        val_loss = torch.mean(loss_func_mse(t_outputs, t_ground_truth))
        val_loss_epoch = val_loss_epoch + val_loss.item()

    val_pred_loss.append(val_loss_epoch)



    print('----------------------------------------')
    print('Epoch:', epoch+1)
    print('Train Loss: Prediction {:.6f}'.format(loss_pixel.item()), 'Validation Loss: Prediction {:.6f}'.format(val_loss.item()))
    print('----------------------------------------')
    np.save(os.path.join(log_dir,'training_loss.npy'),np.asarray(pred_loss))
    np.save(os.path.join(log_dir,'validation_loss.npy'),np.asarray(val_pred_loss))

    torch.save(model.state_dict(), os.path.join(log_dir, 'model_test_tubes.pth'))
 
print('Training is finished')
# Save the model and the memory items

#torch.save(model, os.path.join(log_dir, 'model.pth'))
torch.save(model.state_dict(), os.path.join(log_dir, 'model_test_tubes.pth'))
#torch.save(m_items, os.path.join(log_dir, 'keys.pt'))

plt.plot(pred_loss)
plt.plot(val_pred_loss) 
plt.savefig('/shared/home/v_samveg_shah/local_scratch/plot.png')