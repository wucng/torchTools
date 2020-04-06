# torchTools
some general tools for Classification ,Object detection and Object segmentation GAN and so on

# install
```py
# 方式一
# github上安装
pip install https://github.com/wucng/torchTools/archive/master.zip

# 方式二
# 源码安装
!git clone https://github.com/wucng/torchTools.git
!cd torchTools
!python  setup.py  sdist
!cd sdist
!ls 	# torchTools-0.0.1.tar.gz
!pip install torchTools-0.0.1.tar.gz
!rm -rf torchTools
```
# 使用
```py
from torchTools.classify.dataProcess import loadData,dataAugment
from torchTools.classify.visual import tool
from torchTools.classify.network import net
from torchTools.classify.optimizer import optimizer
from torchTools.classify.loss import loss
from torchTools.classify.learner import ClassifyModel
import numpy as np
import os
```
