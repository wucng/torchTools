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
cfg["network"]["backbone"]["model_name"] = "resnet34"
cfg["network"]["backbone"]["pretrained"] = True
cfg["work"]["train"]["resize"] = resize
cfg["work"]["train"]["epochs"] = 50
cfg["work"]["train"]["classes"] = classes
cfg["work"]["train"]["useImgaug"] = True
cfg["work"]["train"]["version"] = "v2"
cfg["work"]["train"]["method"] = 1
cfg["network"]["backbone"]["freeze_at"] = "res2"
cfg["network"]["RPN"]["num_boxes"] = 6 # 2
cfg["network"]["RPN"]["num_classes"] = len(classes)
cfg["work"]["loss"]["alpha"] = 0.2
cfg["work"]["loss"]["threshold_conf"] = 0.2
cfg["work"]["loss"]["threshold_cls"] = 0.2
cfg["work"]["loss"]["conf_thres"] = 0.4

# """
cfg["network"]["FPN"]["use_FPN"] = True
cfg["network"]["FPN"]["out_features"] = ["p3","p5"]
cfg["network"]["RPN"]["in_channels"] = 256
"""
cfg["network"]["backbone"]["out_features"]=["res5"]
cfg["network"]["FPN"]["use_FPN"] = False
cfg["network"]["RPN"]["in_channels"] = 512
# """
index = cfg["network"]["backbone"]["index"]
if cfg["network"]["FPN"]["use_FPN"]:
    name_features = cfg["network"]["FPN"]["name_features"]
    out_features = cfg["network"]["FPN"]["out_features"]
    for out in out_features:
        index.append(name_features.index(out))
else:
    name_features = cfg["network"]["backbone"]["name_features"]
    out_features = cfg["network"]["backbone"]["out_features"]
    for out in out_features:
        index.append(name_features.index(out))

cfg["network"]["backbone"]["strides"] =[value for i,value in
                            enumerate(cfg["network"]["backbone"]["strides"]) if i in index]
cfg["network"]["prioriBox"]["min_sizes"] = [value for i,value in
                            enumerate(cfg["network"]["prioriBox"]["min_sizes"]) if i in index]
cfg["network"]["prioriBox"]["max_sizes"] = [value for i,value in
                            enumerate(cfg["network"]["prioriBox"]["max_sizes"]) if i in index]
cfg["network"]["prioriBox"]["aspect_ratios"] = [value for i,value in
                            enumerate(cfg["network"]["prioriBox"]["aspect_ratios"]) if i in index]

model = yolo.YOLO(cfg)

# model()
model.predict(5)
# model.eval()

# Look at training curves in tensorboard:
%load_ext tensorboard
%tensorboard --logdir output
```