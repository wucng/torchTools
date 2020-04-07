# 基本步骤
```py
from torchTools.gan.learner_condition import GANModel
# from torchTools.gan.learner import GANModel

num_classes = 10
gan = GANModel(num_classes,base_path="/kaggle/working/")
gan.fit()

import torch

target = torch.randint(0,num_classes,[64],dtype=torch.long,device=torch.device("cuda"))
gan.predict(target)

```

# 自定义数据
```py
# https://www.kaggle.com/alessiocorrado99/animals10

from torchTools.gan.learner_condition import GANModel
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
base_dir = "/kaggle/input/animals10/raw-img/"
image_size = 32

train_transformations = transforms.Compose([
                transforms.Resize((image_size,image_size)),
                transforms.CenterCrop((image_size,image_size)),
                transforms.ToTensor(), # 0~1
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) # -1.0~1.0
            ])
datas = datasets.ImageFolder(base_dir,transform=train_transformations)
num_classes = len(datas.classes)

gan = GANModel(num_classes,
			   image_size=image_size,
			   train_dataset = datas,
			   base_path="/kaggle/working/")
gan.fit()
```

# 自定义模型
```py
# https://www.kaggle.com/alessiocorrado99/animals10

import torch
import torch.nn.functional as F
import torch.nn as nn
from torchTools.gan.learner import GANModel
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset

nc = 3
ndf = 64
nz = 100
ngf = 64
base_dir = "/kaggle/input/animals10/raw-img/"
image_size = 112

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
    def forward(self, x):
        return torch.flatten(x,1)


dnetwork = nn.Sequential(
        nn.Conv2d(nc, ndf, 4, 2, 1, bias=False), # [1, 64, 56, 56]
        nn.LeakyReLU(0.2, inplace=True),

        nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),#[1,128,28,28]
        nn.BatchNorm2d(ndf * 2),
        nn.LeakyReLU(0.2, inplace=True),

        nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),#[1,256,14,14]
        nn.BatchNorm2d(ndf * 4),
        nn.LeakyReLU(0.2, inplace=True),

        nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),#[1,512,7,7]
        nn.BatchNorm2d(ndf * 8),
        nn.LeakyReLU(0.2, inplace=True),

        nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),#[1,1,7,7]
        nn.AdaptiveAvgPool2d((1,1)),
        Flatten(),

        nn.Sigmoid()
)

gnetwork = nn.Sequential(
        nn.ConvTranspose2d(nz, ngf * 8, 7, 1, 0, bias=False),
        nn.BatchNorm2d(ngf * 8),
        nn.ReLU(True),

        nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ngf * 4),
        nn.ReLU(True),

        nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ngf * 2),
        nn.ReLU(True),

        nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ngf),
        nn.ReLU(True),

        nn.ConvTranspose2d( ngf*2, nc, 4, 2, 1, bias=False),
        # nn.Sigmoid(), # 对应图像 norm 0.~1.
        nn.Tanh() # 对应图像norm -1.0~1.0
)


train_transformations = transforms.Compose([
                transforms.Resize((image_size,image_size)),
                transforms.CenterCrop((image_size,image_size)),
                transforms.ToTensor(), # 0~1
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) # -1.0~1.0
            ])
datas = datasets.ImageFolder(base_dir,transform=train_transformations)
num_classes = len(datas.classes)

gan = GANModel(nc=nc,nz=nz,
               image_size=image_size,
			   train_dataset = datas,
			   dnetwork = dnetwork,
			   gnetwork = gnetwork,
			   base_path="/kaggle/working/")
gan.fit()
```