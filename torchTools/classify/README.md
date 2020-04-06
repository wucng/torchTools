# 基本步骤
```py
# https://www.kaggle.com/puneet6060/intel-image-classification
# https://github.com/wucng/torchTools

from torchTools.classify.dataProcess import loadData,dataAugment
from torchTools.classify.visual import tool
from torchTools.classify.network import net
from torchTools.classify.optimizer import optimizer
from torchTools.classify.loss import loss
from torchTools.classify.learner import ClassifyModel
import numpy as np
import os

# 数据路径
# root = "/kaggle/input/animals10/raw-img/"
root = "/kaggle/input/intel-image-classification/"
# 输出路径
base_path = "/kaggle/working/"

# 手动分训练数据与验证数据
# train_datas,valid_datas,classnames=loadData.splitData(root,valid_rote=0.3)

train_datas = loadData.glob_format(os.path.join(root,"seg_train/seg_train"))
valid_datas = loadData.glob_format(os.path.join(root,"seg_test/seg_test"))
classnames = loadData.get_classnames(os.path.join(root,"seg_train/seg_train"))
pred_datasPath = os.path.join(root,"seg_pred/seg_pred")

# 统计每个类别样本分布
tool.static_data(train_datas,classnames)
tool.static_data(valid_datas,classnames)

# 显示部分数据
images,labels = tool.get_img_label(train_datas,classnames,counts=25)
tool.plot_images(images,labels)


# train_transformations,test_transformations = dataAugment.get_transforms((256, 256),(224,224))
# 结合imgaug做数据增强
train_transformations,test_transformations = dataAugment.get_transforms_height((256, 256),(224,224))
train_dataset = loadData.Data_train_valid2(train_datas,classnames,train_transformations)
test_dataset = loadData.Data_train_valid2(valid_datas,classnames,test_transformations)
pred_dataset = loadData.Data_pred(pred_datasPath,test_transformations)
num_classes = len(classnames)

network = net.Resnet(num_classes,'resnet50',True,0.5)
optimizer = optimizer.RAdam

lossFunc = loss.LossFunc(num_classes,reduction="sum").focal_cross_entropy

# 简化版
cls = ClassifyModel(num_classes,
					train_dataset=train_dataset,
					test_dataset=test_dataset,
                    base_path=base_path,
					useTensorboard=False)

# 自定义版
cls = ClassifyModel(num_classes,epochs=2,lr=2e-3,
                    train_dataset=train_dataset,
                    test_dataset=test_dataset,
                    pred_dataset=pred_dataset,
                    network=network,
                    optimizer=optimizer,
                    lossFunc=lossFunc,
                    base_path=base_path,
                    useTensorboard=True)

# 进阶版
cls = ClassifyModel(num_classes,epochs=1,lr=2e-3,
                    train_dataset=train_dataset,
                    test_dataset=test_dataset,
                    pred_dataset=pred_dataset,
                    network=network,
                    optimizer=optimizer,
                    lossFunc=lossFunc, 
                    base_path=base_path,  # 输出文件保存的根目录
                    useTensorboard=False, # 是否使用Tensorboard
					useAdvance=True,      # 是否使用进一步的数据增强（ricap，mixup）
					useParallel=True,     # 是否启用数据并行
					parallels=[0]         # 指定GPU编号
					)

cls.fit()

# 混淆矩阵
y_true, y_pred,features = cls.getTrueAndPred(wantFeature=True)
tool.show_confusion_matrix2(y_true, y_pred,classnames)

# 可视化特征
tool.visual_feature_TSNE(np.asarray(features,np.float32),np.asarray(y_pred,np.int32)[:,0])

# 显示预测数据
imgsPath, labels = cls.predict()
images,_ = tool.get_img_label(imgsPath,counts=25)
tool.plot_images(images,labels)
```