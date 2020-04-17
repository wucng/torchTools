```py
#!pip install https://github.com/wucng/torchTools/archive/master.zip
from torchTools.detection.run import yolov1
from torchTools.detection.datasets import datasets, bboxAug
from torchTools.detection.network import netSmall,netLarger

classes=["person"]
testdataPath = "/kaggle/input/pennfudanped/PNGImages/"
traindataPath = "/kaggle/input/pennfudanped/"
basePath = "/kaggle/working/"
resize = (416,416)
mulScale = False
network = netSmall.YOLOV1Net

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

model = yolov1.YOLOV1(network,train_dataset, test_dataset, "resnet18", pretrained=True, num_features=1,
                   isTrain=True, num_anchors=2, epochs=400, print_freq=40,mulScale=mulScale,
                   basePath=basePath, threshold_conf=0.5, threshold_cls=0.5, lr=3e-3, batch_size=4,
                   conf_thres=0.7, nms_thres=0.4, classes=classes,usize=256)


model()

model.history.show()

model.test(5)
```