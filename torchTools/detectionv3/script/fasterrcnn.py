import os,cv2
import numpy as np
import torch
from torch import nn
import torch.utils.data
from PIL import Image
import torchvision
from tqdm import tqdm
from torchvision import transforms as TT
import matplotlib.pyplot as plt
import time
from torch.utils.tensorboard import SummaryWriter
try:
    from ..tools.engine import train_one_epoch, evaluate
    from ..tools import utils,transforms as T
    # from ..tools.nms_pytorch import nms2 as nms
    from ..network.fasterrcnn import fasterrcnn
    from ..visual import opencv,colormap
    from ..datasets.datasets2 import PennFudanDataset,glob_format,PascalVOCDataset
    from ..datasets import bboxAug
    from ..optm import optimizer
except:
    import sys
    sys.path.append("..")
    from tools.engine import train_one_epoch, evaluate
    from tools import utils, transforms as T
    # from tools.nms_pytorch import nms2 as nms
    from network.fasterrcnn import fasterrcnn
    # from loss import yoloLoss
    from datasets.datasets2 import PennFudanDataset, glob_format,PascalVOCDataset
    from visual import opencv,colormap
    from datasets import bboxAug
    from optm import optimizer


def get_transform(train,advanced=False):
    transforms = []
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        transforms_list = [
            # bboxAug.RandomChoice(),

            # ---------两者取其一--------------------
            bboxAug.RandomHorizontalFlip(),
            bboxAug.RandomTranslate(), # 如果有mask也需相应修改
            # bboxAug.RandomRotate(3),
            bboxAug.RandomBrightness(),
            bboxAug.RandomSaturation(),
            bboxAug.RandomHue(),
            bboxAug.RandomBlur(),
            # ---------两者取其一--------------------
            # bboxAug.Augment(advanced),
            # ---------两者取其一--------------------
        ]

        transforms.extend(transforms_list)

    transforms.append(bboxAug.ToTensor())

    return T.Compose(transforms)

def get_transform2(train):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(T.ToTensor())
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        transforms.append(T.RandomHorizontalFlip(0.5))

    return T.Compose(transforms)


class Fasterrcnn(nn.Module):
    def __init__(self,trainDP=None,classes=[],model_name="resnet101",
                 pretrained=False,out_channels = 256,use_FPN=False,
                 lr=5e-4,num_epochs = 10,filter_labels=[],
                 print_freq=20,rpn_nms_thresh=0.7,
                 box_score_thresh=0.05,
                 box_nms_thresh=0.5,
                 batch_size=2,test_batch_size = 2,
                 basePath="./models/",save_model="model.pt",
                 summaryPath="fasterrcnn",
                 typeOfData = "PennFudanDataset"):
        super(Fasterrcnn,self).__init__()
        # our dataset has two classes only - background and person
        # num_classes = num_classes
        # let's train it for 10 epochs
        self.num_epochs = num_epochs
        self.print_freq = print_freq
        self.test_batch_size = test_batch_size
        self.classes = classes
        num_classes = len(classes)+1 # 加上背景
        self.filter_labels = filter_labels
        self.batch_size = batch_size

        seed = 100
        # seed = int(time.time() * 1000)
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        torch.manual_seed(seed)
        kwargs = {'num_workers': 5, 'pin_memory': True} if self.use_cuda else {}

        if typeOfData == "PennFudanDataset":
            Data = PennFudanDataset
        elif typeOfData == "PascalVOCDataset":
            Data = PascalVOCDataset
        else:
            pass
        # use our dataset and defined transformations
        dataset = Data(trainDP, transforms=get_transform(train=True),classes = classes)
        dataset_test = Data(trainDP, transforms=get_transform(train=False),classes = classes)

        # split the dataset in train and test set
        # torch.manual_seed(1)
        num_datas = len(dataset)
        indices = torch.randperm(num_datas).tolist()
        dataset = torch.utils.data.Subset(dataset, indices[:int(num_datas*0.8)])
        dataset_test = torch.utils.data.Subset(dataset_test, indices[int(num_datas*0.8):])

        # define training and validation data loaders
        self.train_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True,
            collate_fn=utils.collate_fn,**kwargs)

        self.test_loader = torch.utils.data.DataLoader(
            dataset_test, batch_size=batch_size//2 if batch_size//2 else 1, shuffle=False,
            collate_fn=utils.collate_fn,**kwargs)


        self.network = fasterrcnn.FasterRCNN(model_name=model_name,pretrained=pretrained,
                                             out_channels=out_channels,useFPN=use_FPN,
                                             num_classes=num_classes,rpn_nms_thresh=rpn_nms_thresh,
                                             box_score_thresh=box_score_thresh,
                                             box_nms_thresh=box_nms_thresh
                                             )

        if self.use_cuda:
            # move model to the right device
            self.network.to(self.device)

        if not os.path.exists(basePath):
            os.makedirs(basePath)

        self.save_model = os.path.join(basePath,save_model)
        if os.path.exists(self.save_model):
            self.network.load_state_dict(torch.load(self.save_model))

        # """
        # construct an optimizer
        params = [p for p in self.network.parameters() if p.requires_grad]
        self.optimizer = torch.optim.SGD(params, lr=lr,momentum=0.9, weight_decay=5e-4)

        # and a learning rate scheduler which decreases the learning rate by
        # 10x every 3 epochs
        # self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,step_size=3,gamma=0.1)
        self.lr_scheduler = optimizer.build_lr_scheduler(self.optimizer)
        """
        self.optimizer = optimizer.build_optimizer(self.network,lr,clip_gradients=True)
        self.lr_scheduler = optimizer.build_lr_scheduler(self.optimizer)

        self.writer = SummaryWriter(os.path.join(basePath, summaryPath))
        # """
    def forward(self):
        for epoch in range(self.num_epochs):
            self.train(epoch)
            # self.train2(epoch)
            # update the learning rate
            self.lr_scheduler.step()
            # self.eval()
            torch.save(self.network.state_dict(), self.save_model)


    def train2(self, epoch):
        self.network.train()
        num_trains = len(self.train_loader.dataset)
        for idx, (data, target) in enumerate(self.train_loader):
            if self.use_cuda:
                data = list(da.to(self.device) for da in data)
                target = [{k: v.to(self.device) for k, v in targ.items()} for targ in target]

            loss_dict = self.network(data, target)

            losses = sum(loss for loss in loss_dict.values())

            self.optimizer.zero_grad()
            losses.backward()
            self.optimizer.step()

            # 记录到TensorBoard
            self.writer.add_scalar('total_loss', losses.item(), epoch * num_trains // self.batch_size + idx)
            for key, loss in loss_dict.items():
                self.writer.add_scalar(key, loss.item(), epoch * num_trains // self.batch_size + idx)

            if idx % self.print_freq == 0:
                ss = "epoch:{}-({}/{})".format(epoch, idx * self.batch_size, num_trains)
                ss += "\ttotal:{:.3f}".format(losses.item())
                for key, loss in loss_dict.items():
                    ss += "\t{}:{:.3f}".format(key, loss.item())

                print(ss)

    def train(self,epoch):
            # train for one epoch, printing every 10 iterations
            train_one_epoch(self.network, self.optimizer, self.train_loader, self.device, epoch, self.print_freq)

    def eval(self):
        # evaluate on the test dataset
        evaluate(self.network, self.test_loader, self.device)

    def predict(self,path,nums=None):
        self.network.eval()
        filenames = glob_format(path)
        preprocess = TT.Compose([
            TT.ToTensor()
        ])
        # for i in tqdm(range(len(filenames) // self.test_batch_size)):
        for i in range(len(filenames) // self.test_batch_size):
            if nums is not None and i>nums:break
            temp = filenames[i * self.test_batch_size:(i + 1) * self.test_batch_size]
            input_batch = [preprocess(Image.open(filename).convert("RGB")).to(self.device) for filename in temp]

            with torch.no_grad():
                detections = self.network(input_batch)

            for idx, filename in enumerate(temp):
                # image = cv2.imread(filename)
                image = np.asarray(Image.open(filename).convert("RGB"), np.uint8)
                _detections = detections[idx]
                if _detections is None or len(_detections)==0:
                    # cv2.imwrite(filename.replace("image","out"),image)
                    continue
                image = self.draw_rect(image, _detections)
                cv2.imshow("test", image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

                # plt.imshow(image)
                # plt.show()

                # save
                # newPath = path.replace("PNGImages", "result")
                # if not os.path.exists(os.path.dirname(newPath)): os.makedirs(os.path.dirname(newPath))
                # cv2.imwrite(newPath, image)

    def draw_rect(self,image, pred):
        """segms=False 不画mask"""
        labels = pred["labels"]
        bboxs = pred["boxes"]
        scores = pred["scores"]

        for idx,(label, bbox, score) in enumerate(zip(labels, bboxs, scores)):
            label = label.cpu().numpy()
            bbox = bbox.cpu().numpy()  # .astype(np.int16)
            score = score.cpu().numpy()
            class_str = "%s:%.3f" % (self.classes[int(label)-1], score)  # 跳过背景
            pos = list(map(int, bbox))
            image = opencv.vis_rect(image, pos, class_str, 0.5, int(label),useMask=True)

        return image

if __name__ == "__main__":
    classes = ["person"]
    testdataPath = r"D:\practice\datas\PennFudanPed\PNGImages"
    traindataPath = r"D:\practice\datas\PennFudanPed"
    # testdataPath = "../../datas/PennFudanPed/PNGImages"
    # traindataPath = "../../datas/PennFudanPed"
    basePath = "./models"
    typeOfData = "PennFudanDataset"

    model = Fasterrcnn(traindataPath, classes, "resnet18", pretrained=True,
                       out_channels=256,use_FPN=True,lr=5e-3,num_epochs=10,
                       print_freq=20,box_score_thresh=0.3,box_nms_thresh=0.4,
                       batch_size=2,basePath=basePath,typeOfData=typeOfData)

    # model()
    model.predict(testdataPath)

    """
    import torch
    from PIL import Image
    
    model.network.eval()
    img,_ = model.test_loader.dataset[0]
    with torch.no_grad():
        detections = model.network([img.to("cuda")])
        
    detections
    
    Image.fromarray(img.mul(255).permute(1, 2, 0).byte().numpy())
    mask = Image.fromarray(detections[0]['masks'][0, 0].mul(255).byte().cpu().numpy())
    
    # let's adda color palette to the mask.
    mask.putpalette([
        0, 0, 0, # black background
        255, 0, 0, # index 1 is red
        255, 255, 0, # index 2 is yellow
        255, 153, 0, # index 3 is orange
    ])
    
    mask
    """