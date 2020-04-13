```py
#!pip install https://github.com/wucng/torchTools/archive/master.zip
from torchTools.detection.run import yolov1

classes=["__background__","person"]
testdataPath = "/kaggle/input/pennfudanped/PNGImages/"
traindataPath = "/kaggle/input/pennfudanped/"
basePath = "/kaggle/working/"
model = yolov1.YOLOV1(traindataPath,testdataPath,"resnet34",pretrained=True,
                      isTrain=True,num_anchors=2,num_classes=1,mulScale=True,epochs=200,
                     basePath = basePath,threshold_conf=0.2,threshold_cls=0.5,
                     conf_thres=0.7,nms_thres=0.4,classes=classes)


model = yolov1.YOLOV1(traindataPath,testdataPath,"resnet34",pretrained=True,num_features=1,
                      isTrain=True,num_anchors=2,num_classes=1,mulScale=False,epochs=200,
                     basePath = basePath,threshold_conf=0.2,threshold_cls=0.5,
                     conf_thres=0.7,nms_thres=0.4,classes=classes)


model = yolov1.YOLOV1(traindataPath,testdataPath,"resnet50",pretrained=True,num_features=None,
                      isTrain=True,num_anchors=2,num_classes=1,mulScale=False,epochs=400,print_freq=40,
                     basePath = basePath,threshold_conf=0.2,threshold_cls=0.5,lr=3e-3,batch_size=2,
                     conf_thres=0.7,nms_thres=0.4,classes=classes)

model()
```