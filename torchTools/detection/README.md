```py
#!pip install https://github.com/wucng/torchTools/archive/master.zip
from torchTools.detection.run import yolov1
from torchTools.detection.datasets import datasets, bboxAug

classes=["person"]
testdataPath = "/kaggle/input/pennfudanped/PNGImages/"
traindataPath = "/kaggle/input/pennfudanped/"
basePath = "/kaggle/working/"
resize = (416,416)
mulScale = False

train_transforms=bboxAug.Compose([
    bboxAug.Pad(), bboxAug.Resize(resize, mulScale),
    bboxAug.ToTensor(),
    bboxAug.Normalize()
])

test_transforms=bboxAug.Compose([
    bboxAug.Pad(), bboxAug.Resize(resize, mulScale),
    bboxAug.ToTensor(),
    bboxAug.Normalize()
])

train_dataset = datasets.PennFudanDataset(traindataPath,train_transforms,classes)
test_dataset = datasets.ValidDataset(testdataPath,test_transforms)

model = yolov1.YOLOV1(train_dataset, test_dataset, "resnet18", pretrained=True, num_features=1,
                   isTrain=True, num_anchors=2, epochs=400, print_freq=40,
                   basePath=basePath, threshold_conf=0.5, threshold_cls=0.5, lr=3e-3, batch_size=4,
                   conf_thres=0.7, nms_thres=0.4, classes=classes,typeOfData=typeOfData,usize=256)


model()
```