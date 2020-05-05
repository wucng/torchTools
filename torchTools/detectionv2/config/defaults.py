_cfg = {
    "network":{
        "backbone":{
            "model_name":"resnet18",
            "pretrained":False,
            "dropRate":0.5,
            "freeze_at":"res2",
            "name_features":["res2","res3","res4","res5"],
            "out_features":["res2","res3","res4","res5"],
            # "out_features":["res5"],
            "strides":[4,8,16,32],
            "index":[]
        },
        "FPN":{
            "use_FPN":True,
            "name":"FPNNet", # "FPNNetCH","FPNNetLarger","FPNNetSmall","XNet"
            "usize":256,
            "name_features":["p2","p3","p4","p5"],
            "out_features":["p2","p3","p4","p5"],
            # "out_features":["p3"]
        },
        "RPN":{
            "in_channels":256,
            "num_boxes":2,
            "num_classes":1, # 不包括背景

        },
        "proposal":{
            "threshold_cls":0.05,
            "nms_thres":0.7,
            "num_anchor":1000
        },
        "RCNN":{
            "in_channels":256,
            # "num_boxes":1,
            "num_classes":1, # 不包括背景
            "align_features":"p3",
            "thred_iou":0.7
        },

        # ssd对应得先验框
        "prioriBox":{
            "min_dim":300,
            "min_sizes":[30,60,111,162],
            "max_sizes":[60,111,162,213],
            "aspect_ratios":[[2,3],[2,3],[2,3],[2,3]], #[[2],[2],[2],[2]]
            "variance":[0.1,0.2],
            "clip":True,
            "thred_iou":0.5
        }
    },
    "work":{
        "dataset": {
            "trainDataPath":None,
            "testDataPath":None,
            "predDataPath":None,
            "typeOfData":"PennFudanDataset"
        },
        "save":{
            "basePath":"./",
            "save_model":"model.pt",
            "summaryPath":"yolov1"
        },
        "train":{
            "isTrain":True,
            "batch_size":2,
            "epochs":100,
            "print_freq":50,
            "resize":(224,224),
            "advanced":False,
            "useImgaug":True,
            # "lr":5e-4,
            "train_method":1,
            "method":0,

            "classes":[],
            "filter_labels":[],
            "version":"v1"
        },
        "optimizer":{
            "clip_gradients":True,
            "base_lr":2.5e-4, # 2.5e-4
            "weight_decay":1e-4,
            "momentum":0.9,
            "clip_type":"value", # norm
            "lr_scheduler_name":"WarmupMultiStepLR",# "WarmupCosineLR"

        },
        "loss":{
            "useFocal":True,
            "alpha":0.2,#0.4
            "gamma":2,
            "threshold_conf":0.3,
            "threshold_cls":0.3,
            "conf_thres":0.3,
            "nms_thres":0.4,
        }
    }
}