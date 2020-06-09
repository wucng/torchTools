from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
from glob import glob
import PIL.Image
import os
import torch
import cv2

def get_classnames(base_dir):
    classnames = sorted(os.listdir(os.path.join(base_dir)))
    return classnames

# load data
def glob_format(path, base_name=False):
    # print('--------pid:%d start--------------' % (os.getpid()))
    fmt_list = ('.jpg', '.jpeg', '.png')
    fs = []
    if not os.path.exists(path): return fs
    for root, dirs, files in os.walk(path):
        for file in files:
            item = os.path.join(root, file)
            # item = unicode(item, encoding='utf8')
            fmt = os.path.splitext(item)[-1]
            if fmt.lower() not in fmt_list:
                # os.remove(item)
                continue
            if base_name:
                fs.append(file)  # fs.append(os.path.splitext(file)[0])
            else:
                fs.append(item)
    # print('--------pid:%d end--------------' % (os.getpid()))
    return fs


# read .txt 结合命令: tree -i -f `pwd`/20181022/json_data/|grep .jpg >data.txt
def read_txt(path):#, batch_frame_queue, batch_size):
    # print('--------pid:%d start--------------' % (os.getpid()))
    batch_frames_info = []
    fp = open(path)
    try:
        for line in fp:
            if line[-5:] == ".jpg\n":
                batch_frames_info.append(line.strip())
            # if len(batch_frames_info) == batch_size:
            #     batch_frame_queue.put(batch_frames_info)
            #     batch_frames_info = []

    except Exception as e:
        print(e)
        fp.close()
        # break

    # if len(batch_frames_info) > 0:
    #     batch_frame_queue.put(batch_frames_info)
    # print('--------pid:%d end--------------' % (os.getpid()))
    return batch_frames_info


# 用于predict，没有标签
class Data_pred(Dataset):
    def __init__(self,root,transform=None):
        super(Data_pred, self).__init__()
        # self.datas = glob(os.path.join(root,"*.jpg"))
        self.datas = glob_format(root)
        self.transform = transform

    def __len__(self):
        return len(self.datas)

    def __getitem__(self,idx):
        path = self.datas[idx]
        img = PIL.Image.open(path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        return img,path


"""
文件夹对应类别名
xxx/other/12.jpg
xxx/side_face/23.jpg
xxx/positive_face/23.jpg

# or 使用内置方法
datas = datasets.ImageFolder(os.path.join(base_dir,"seg_train/seg_train"),transform=train_transformations)
datas.classes
"""
class Data_train_valid(Dataset):
    def __init__(self, base_dir, mode="train", transform=None, target_transform=None, shuffle=False):
        super(Data_train_valid, self).__init__()
        self.paths = glob_format(os.path.join(base_dir, mode))

        if shuffle: self._shuttle()

        # self.classes=classes
        self.classnames = sorted(os.listdir(os.path.join(base_dir, mode)))
        self.transform = transform
        self.target_transform = target_transform

    def _shuttle(self):
        # np.random.shuffle(self.paths)
        random.shuffle(self.paths)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = PIL.Image.open(path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        # target
        target = self.classnames.index(os.path.basename(os.path.dirname(path)))

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


# 只有训练数据没有分验证数据，手动拆分
# split train datas and valid datas
def splitData(root,valid_rote = 0.3):
    classnames = sorted(os.listdir(root))
    train_datas = []
    valid_datas = []
    for name in classnames:
        path = os.path.join(root,name)
        # for img in os.listdir(path):
        # imgs = glob(os.path.join(path,"*"))
        imgs = glob_format(path)
        # shuffle
        # np.random.shuffle(imgs)
        random.shuffle(imgs)
        len_tdatas = int(len(imgs)*valid_rote)
        valid_datas.extend(imgs[:len_tdatas])
        train_datas.extend(imgs[len_tdatas:])

    return train_datas,valid_datas,classnames

class Data_train_valid2(Dataset):
    def __init__(self,datas,classnames,transform=None,target_transform=None):
        super(Data_train_valid2, self).__init__()
        self.datas = datas
        self.classnames = classnames
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.datas)

    def _load(self):
        path = self.datas[idx]
        img = PIL.Image.open(path).convert("RGB")
        # target
        target = self.classnames.index(os.path.basename(os.path.dirname(path)))
        return img,target

    def __getitem__(self,idx):
        img, target = self._load()
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


# 2.ricap, 3.mixup 数据增强
class Data_train_valid3(Dataset):
    def __init__(self,datas,classnames,transform=None,target_transform=None):
        super(Data_train_valid3, self).__init__()
        self.datas = datas
        self.classnames = classnames
        self.transform = transform
        self.target_transform = target_transform
        self.n_classes = len(self.classnames)
        self.alpha = 0.8
        self.alpha_ricap = 0.7
        self.alpha_general = 0.03

    def __len__(self):
        return len(self.datas)

    def _load(self,idx):
        path = self.datas[idx]
        img = np.asarray(PIL.Image.open(path).convert("RGB"),np.uint8)
        # target
        target = self.classnames.index(os.path.basename(os.path.dirname(path)))
        # to onehot
        target = np.eye(self.n_classes,self.n_classes)[target]

        return img,target

    def _mixup(self,idx):
        index = torch.randperm(self.__len__()).tolist()
        if idx + 1 >= self.__len__():
            idx = 0
        idx2 = index[idx + 1]
        img, target = self._load(idx)
        img2, target2 = self._load(idx2)

        # mixup
        img = np.clip(cv2.addWeighted(img,self.alpha,img2,1-self.alpha,gamma=0.0),0,255).astype(np.uint8)
        target = target*self.alpha+target2*(1-self.alpha)

        return img,target


    def _ricap(self,idx):
        # 类似Mosaic数据增强
        index = torch.randperm(self.__len__()).tolist()
        if idx + 3 >= self.__len__():
            idx = 0
        idx2 = index[idx + 1]
        idx3 = index[idx + 2]
        idx4 = index[idx + 3]

        img, target = self._load(idx)
        img2, target2 = self._load(idx2)
        img3, target3 = self._load(idx3)
        img4, target4 = self._load(idx4)

        h1, w1, _ = img.shape
        h2, w2, _ = img2.shape
        h3, w3, _ = img3.shape
        h4, w4, _ = img4.shape

        h = max((h1, h2, h3, h4))
        w = max((w1, w2, w3, h4))

        th=int(h*self.alpha_ricap)
        tw=int(w*self.alpha_ricap)

        temp_img = np.zeros((h,w,3),np.uint8)
        temp_img[:th,:tw]=cv2.resize(img,(tw,th),interpolation=cv2.INTER_BITS)
        temp_img[:th,tw:]=cv2.resize(img2,(w-tw,th),interpolation=cv2.INTER_BITS)
        temp_img[th:,:tw]=cv2.resize(img3,(tw,h-th),interpolation=cv2.INTER_BITS)
        temp_img[th:,tw:]=cv2.resize(img4,(w-tw,h-th),interpolation=cv2.INTER_BITS)

        target = (target*(tw*th)+target2*(w-tw)*th+target3*tw*(h-th)+target4*(w-tw)*(h-th))/(h*w)

        return temp_img,target


    def __getitem__(self,idx):
        state = np.random.choice(["general", "ricap", "mixup"], 1)[0]
        if state == "general":
            img, target = self._load(idx)
            # smooth label
            target = target * (1 - self.alpha_general) + self.alpha_general / self.n_classes * np.ones_like(target)
        elif state == "ricap":
            img, target = self._ricap(idx)
        else:
            img, target = self._mixup(idx)

        img = PIL.Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target