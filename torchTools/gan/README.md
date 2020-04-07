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

# 新数据
```py
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