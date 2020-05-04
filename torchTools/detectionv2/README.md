- [install](#install)
- [yolo](#yolo)

---
# install
```python
# !pip install -U https://github.com/wucng/torchTools/archive/master.zip   
!pip install -U git+https://github.com/wucng/torchTools.git  
!pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
!pip install git+https://github.com/facebookresearch/fvcore.git
```

# yolo
```python
from torchTools.detectionv2.script import yolo
from torchTools.detectionv2.config import config


classes=["person"]
preddataPath = "/kaggle/input/pennfudanped/PNGImages/"
testdataPath = None
traindataPath = "/kaggle/input/pennfudanped/"
basePath = "/kaggle/working/modelsv1"
resize = (416,416)
typeOfData = "PennFudanDataset"

cfg = config.get_cfg()
cfg["work"]["dataset"]["trainDataPath"] = traindataPath
cfg["work"]["dataset"]["testDataPath"] = testdataPath
cfg["work"]["dataset"]["predDataPath"] = preddataPath
cfg["work"]["dataset"]["typeOfData"] = typeOfData
cfg["work"]["save"]["basePath"] = basePath
cfg["network"]["backbone"]["model_name"]="resnet34"
cfg["network"]["backbone"]["pretrained"]=True
cfg["work"]["train"]["resize"]=resize
cfg["work"]["train"]["epochs"]=50
cfg["work"]["train"]["classes"]=classes
cfg["work"]["train"]["useImgaug"]=False
cfg["network"]["backbone"]["freeze_at"]="res2"
cfg["network"]["RPN"]["num_boxes"]=2
cfg["network"]["RPN"]["num_classes"]=len(classes)
cfg["work"]["loss"]["alpha"]=0.4
cfg["work"]["loss"]["threshold_conf"]=0.2
cfg["work"]["loss"]["threshold_cls"]=0.2
cfg["work"]["loss"]["conf_thres"]=0.2

"""
cfg["network"]["backbone"]["strides"] = [8]
cfg["network"]["FPN"]["use_FPN"] = True
cfg["network"]["FPN"]["out_features"] = ["p3"]
cfg["network"]["RPN"]["in_channels"] = 256
"""
cfg["network"]["backbone"]["out_features"]=["res5"]
cfg["network"]["backbone"]["strides"] = [32]
cfg["network"]["FPN"]["use_FPN"] = False
cfg["network"]["RPN"]["in_channels"] = 512
# """

model = yolo.YOLO(cfg)

# model()
model.predict(5)
# model.eval()

# Look at training curves in tensorboard:
%load_ext tensorboard
%tensorboard --logdir output
```