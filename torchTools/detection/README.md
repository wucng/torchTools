```py
!pip install -U https://github.com/wucng/torchTools/archive/master.zip
#or !pip install 'git+https://github.com/wucng/torchTools.git'
!pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
!pip install git+https://github.com/facebookresearch/fvcore.git

from torchTools.detection.script import cascadessd,ssd,yolo,fasterrcnn

classes=["person"]
preddataPath = "/kaggle/input/pennfudanped/PNGImages/"
testdataPath = "/kaggle/input/pennfudanped/"
traindataPath = "/kaggle/input/pennfudanped/"
basePath = "/kaggle/working/models"
resize = (480,480)
mulScale = False
typeOfData = "PennFudanDataset"

model = yolo.YOLO(traindataPath, testdataPath,preddataPath,"resnet34", pretrained=True, num_features=1,resize=resize,
                   isTrain=True, num_anchors=2, mulScale=mulScale, epochs=50, print_freq=50,dropRate=0.0,
                   basePath=basePath, threshold_conf=0.5, threshold_cls=0.5, lr=5e-4, batch_size=2,freeze_at=0,
                   conf_thres=0.7, nms_thres=0.4, classes=classes,typeOfData=typeOfData,usize=256,version="v1",
                 useFocal=True,train_method=1)

model = ssd.SSD(traindataPath, testdataPath,preddataPath, "resnet34", pretrained=True, num_features=1,resize=resize,
                   isTrain=True, num_anchors=3, mulScale=mulScale, epochs=50, print_freq=40,dropRate=0.0,
                   basePath=basePath, threshold_conf=0.5, threshold_cls=0.5, lr=3e-4, batch_size=2,
                   conf_thres=0.7, nms_thres=0.4, classes=classes,typeOfData=typeOfData,usize=256,freeze_at=0,
				   useFocal=True,clip=False,train_method=1)

model = fasterrcnn.Fasterrcnn(traindataPath, classes, "resnet50", pretrained=True, num_epochs=10,
                       conf_thres=0.4, nms_thres=0.4, batch_size=2,usize=256,lr=5e-3,
                       use_FPN=True, basePath=basePath, useMask=True, selfmodel=False)

model()

model.eval()
model.predict(5)
model.predict(testdataPath,10)
```
----

```py
#!pip install https://github.com/wucng/torchTools/archive/master.zip
from torchTools.detection.run import yolo
from torchTools.detection.datasets import datasets, bboxAug
from torchTools.detection.network import yoloNet

classes=["person"]
testdataPath = "/kaggle/input/pennfudanped/PNGImages/"
traindataPath = "/kaggle/input/pennfudanped/"
basePath = "/kaggle/working/"
resize = (416,416)
mulScale = False
# network = netSmall.YOLOV1Net

train_transforms=bboxAug.Compose([
    # bboxAug.RandomChoice(),
    bboxAug.Pad(), bboxAug.Resize(resize, mulScale),
    # --------------------------------------
    # bboxAug.RandomHorizontalFlip(),
    # bboxAug.RandomTranslate(),
    # # bboxAug.RandomRotate(3),
    # bboxAug.RandomBrightness(),
    # bboxAug.RandomSaturation(),
    # bboxAug.RandomHue(),
    # bboxAug.RandomBlur(),
    # ---------两者取其一--------------------
    # bboxAug.Augment(advanced),
    # --------------------------------------
    bboxAug.ToTensor(),
    bboxAug.Normalize()
])

test_transforms=bboxAug.Compose([
    bboxAug.Pad(), bboxAug.Resize(resize, False),
    bboxAug.ToTensor(),
    bboxAug.Normalize()
])

train_dataset = datasets.PennFudanDataset(traindataPath,train_transforms,classes)
test_dataset = datasets.ValidDataset(testdataPath,test_transforms)

model = yolo.YOLO(network,train_dataset, test_dataset, "resnet18", pretrained=True, num_features=1,
                   isTrain=True, num_anchors=2, epochs=400, print_freq=40,mulScale=mulScale,
                   basePath=basePath, threshold_conf=0.5, threshold_cls=0.5, lr=3e-3, batch_size=4,
                   conf_thres=0.7, nms_thres=0.4, classes=classes,usize=256,version="v1")


model()

model.history.show()

model.test(5)
```