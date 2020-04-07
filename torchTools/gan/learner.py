"""
普通GAN
"""
from __future__ import print_function
import os
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils import clip_grad_norm_
from torchvision import datasets, transforms
import torchvision
# from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter
import PIL.Image
from collections import OrderedDict
import math
import matplotlib.pyplot as plt
import json

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
    def forward(self, x):
        return torch.flatten(x,1)


# 自定义模型
class Discriminator(nn.Module):
    def __init__(self, nc = 3,ndf = 64):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (ndf*4) x 8 x 8
            # nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(ndf * 8),
            # nn.LeakyReLU(0.2, inplace=True),

            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 4, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

class Generator(nn.Module):
    def __init__(self,nc=3, nz = 100,ngf = 64):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            # state size. (ngf*2) x 16 x 16
            # nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(ngf),
            # nn.ReLU(True),

            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( ngf*2, nc, 4, 2, 1, bias=False),
            # nn.Sigmoid(), # 对应图像 norm 0.~1.
            nn.Tanh() # 对应图像norm -1.0~1.0

            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class GANModel(nn.Module):
    def __init__(self,nc=3,nz=100,epochs=10,droprate=0.5,lr=2e-4,image_size=32,
                 batch_size=32,test_batch_size=64,log_interval=30,
                 train_dataset=None,dnetwork=None,gnetwork=None,
                 optimizer=None,lossFunc=None,
                 base_path="./",save_model="model.pt"):
        """parallels:GPU编号"""
        super(GANModel,self).__init__()
        self.epochs = epochs
        self.droprate = droprate
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.log_interval = log_interval
        seed = 100
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        torch.manual_seed(seed)
        kwargs = {'num_workers': 5, 'pin_memory': True} if self.use_cuda else {}
        self.base_path = base_path
        self.image_size = image_size
        self.nz = nz
        self.nc = nc

        self.train_dataset = train_dataset
        if train_dataset is not None:
            self.train_loader = DataLoader(train_dataset,batch_size=self.batch_size,shuffle=True,**kwargs)
        else:
            self.nc = 1
            self.train_loader = DataLoader(datasets.FashionMNIST(self.base_path,train=True, download=True,transform=transforms.Compose([
                transforms.Resize((self.image_size,self.image_size)),
                transforms.CenterCrop((self.image_size,self.image_size)),
                transforms.ToTensor(), # 0~1
                transforms.Normalize((0.5,), (0.5,)) # -1.0~1.0
                # transforms.Normalize((0.1307,), (0.3081,))
            ])),batch_size=self.batch_size,shuffle=True, **kwargs)

        if dnetwork is None:
            self.dnetwork = Discriminator(self.nc)
        else:
            self.dnetwork = dnetwork

        if gnetwork is None:
            self.gnetwork = Generator(self.nc,nz=self.nz)
        else:
            self.gnetwork = gnetwork

        #  to mean=0, stdev=0.2.
        self.dnetwork.apply(weights_init)
        self.gnetwork.apply(weights_init)

        if self.use_cuda:
            self.dnetwork.to(self.device)
            self.gnetwork.to(self.device)

        if lossFunc is None:
            self.lossFunc = nn.BCELoss(reduction='mean')
        else:
            self.lossFunc = lossFunc

        if optimizer is None:
            self.doptimizer = torch.optim.Adam(self.dnetwork.parameters(), lr=lr, weight_decay=4e-05,
                                               betas=(0.5, 0.999))  # 4e-05
            self.goptimizer = torch.optim.Adam(self.gnetwork.parameters(), lr=lr, weight_decay=4e-05,
                                               betas=(0.5, 0.999))  # 4e-05
        else:
            self.doptimizer = optimizer(self.dnetwork.parameters(), lr=lr, weight_decay=4e-05,betas=(0.5, 0.999))  # 4e-05
            self.goptimizer = optimizer(self.gnetwork.parameters(), lr=lr, weight_decay=4e-05,betas=(0.5, 0.999))  # 4e-05

        self.lr_scheduler_d = torch.optim.lr_scheduler.StepLR(self.doptimizer, step_size=20, gamma=0.1)
        self.lr_scheduler_g = torch.optim.lr_scheduler.StepLR(self.goptimizer, step_size=20, gamma=0.1)

        self.save_dmodel = os.path.join(base_path,"d_"+save_model)
        self.save_gmodel = os.path.join(base_path,"g_"+save_model)

        if os.path.exists(self.save_dmodel):
            self.dnetwork.load_state_dict(torch.load(self.save_dmodel))
        if os.path.exists(self.save_gmodel):
            self.gnetwork.load_state_dict(torch.load(self.save_gmodel))


    # Train the network for one or more epochs, validating after each epoch.
    def train(self, epoch):
        self.dnetwork.train()
        self.gnetwork.train()

        num_trains = len(self.train_loader.dataset)

        if self.train_dataset is not None:
            for batch, (data,) in enumerate(self.train_loader):
                if self.use_cuda:
                    data = data.to(self.device)

                ############################
                # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                ###########################

                b_size = data.size(0)
                real_label = torch.full((b_size,), 1, device=self.device, requires_grad=False)
                fake_label = torch.full((b_size,), 0, device=self.device, requires_grad=False)

                self.dnetwork.zero_grad()
                # or
                # self.doptimizer.zero_grad()
                fixed_noise = torch.randn(b_size, self.nz, 1, 1, device=self.device)
                fake = self.gnetwork(fixed_noise)
                fout = self.dnetwork(fake.detach())
                tout = self.dnetwork(data)
                d_loss = self.lossFunc(tout, real_label) + \
                         self.lossFunc(fout,fake_label)

                d_loss.backward()
                self.doptimizer.step()

                ############################
                # (2) Update G network: maximize log(D(G(z)))
                ###########################
                self.gnetwork.zero_grad()
                # or
                # self.goptimizer.zero_grad()
                fout = self.dnetwork(fake)
                g_loss = self.lossFunc(fout, real_label)+0.2*F.mse_loss(fout,data)
                g_loss.backward()

                # Update G
                self.goptimizer.step()

                if batch % self.log_interval == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tDLoss: {:.6f} GLoss: {:.6f}'.format(epoch, batch * len(data),
                                                                                                  num_trains,
                                                                                                  100. * batch / num_trains,
                                                                                                  d_loss.data.item(),
                                                                                                  g_loss.data.item()))
        else:
            for batch, (data, target) in enumerate(self.train_loader):
                if self.use_cuda:
                    data, target = data.to(self.device), target.to(self.device)

                ############################
                # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                ###########################

                b_size = data.size(0)
                real_label = torch.full((b_size,), 1, device=self.device, requires_grad=False)
                fake_label = torch.full((b_size,), 0, device=self.device, requires_grad=False)

                self.dnetwork.zero_grad()
                # or
                # self.doptimizer.zero_grad()
                fixed_noise = torch.randn(b_size, self.nz, 1, 1, device=self.device)
                fake = self.gnetwork(fixed_noise)
                fout = self.dnetwork(fake.detach())
                tout = self.dnetwork(data)
                d_loss = self.lossFunc(tout, real_label) + \
                         self.lossFunc(fout,fake_label)

                d_loss.backward()
                self.doptimizer.step()

                ############################
                # (2) Update G network: maximize log(D(G(z)))
                ###########################
                self.gnetwork.zero_grad()
                # or
                # self.goptimizer.zero_grad()
                fout = self.dnetwork(fake)
                g_loss = self.lossFunc(fout, real_label)+0.2*F.mse_loss(fout,data)
                g_loss.backward()

                # Update G
                self.goptimizer.step()

                if batch % self.log_interval == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tDLoss: {:.6f} GLoss: {:.6f}'.format(epoch, batch * len(data),
                                                                                                  num_trains,
                                                                                                  100. * batch / num_trains,
                                                                                                  d_loss.data.item(),
                                                                                                  g_loss.data.item()))

    def predict(self):
        # if os.path.exists(self.save_gmodel):
        #     self.gnetwork.load_state_dict(torch.load(self.save_gmodel))
        self.gnetwork.eval()
        with torch.no_grad():
            fixed_noise = torch.randn(64, self.nz, 1, 1, device=self.device)  # 随机噪声数据
            decode = self.gnetwork(fixed_noise)

            # decode得到的图片
            # decode = torch.clamp(decode * 255, 0, 255)  # 对应0~1
            decode = torch.clamp((decode * 0.5 + 0.5) * 255, 0, 255) # -1.0~1.0
            decode = decode.detach().cpu().numpy().astype(np.uint8)
            decode = np.transpose(decode, [0, 2, 3, 1])

            imgs = np.zeros([8 * self.image_size, 8 * self.image_size, self.nc], np.uint8)
            for i in range(8):
                for j in range(8):
                    imgs[i * self.image_size:(i + 1) * self.image_size, j * self.image_size:(j + 1) * self.image_size] = \
                    decode[i * 8 + j]

            # 保存
            # PIL.Image.fromarray(imgs.squeeze(-1)).show()
            # PIL.Image.fromarray(imgs.squeeze(-1)).save("test.jpg")
            # print("保存成功")
            if self.nc == 1:
                plt.imshow(imgs.squeeze(-1), cmap="gray")
            else:
                plt.imshow(imgs, cmap=None)
            # plt.imshow(imgs.squeeze(-1),cmap="gray" if self.nc==1 else None)
            plt.show()

    def fit(self):
        for e in range(self.epochs):
            self.train(e+1)
            self.predict()

            torch.save(self.dnetwork.state_dict(), self.save_dmodel)  # save models
            torch.save(self.gnetwork.state_dict(), self.save_gmodel)  # save models

            # update the learning rate
            self.lr_scheduler_d.step()
            self.lr_scheduler_g.step()