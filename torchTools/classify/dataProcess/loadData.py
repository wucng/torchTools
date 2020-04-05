from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from glob import glob
import PIL.Image
import os

# load data
def glob_format(path, base_name=False):
    print('--------pid:%d start--------------' % (os.getpid()))
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
    print('--------pid:%d end--------------' % (os.getpid()))
    return fs


# read .txt 结合命令: tree -i -f `pwd`/20181022/json_data/|grep .jpg >data.txt
def read_txt(path):#, batch_frame_queue, batch_size):
    print('--------pid:%d start--------------' % (os.getpid()))
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
    print('--------pid:%d end--------------' % (os.getpid()))
    return batch_frames_info


# 用于predict，没有标签
class Data_pred(Dataset):
    def __init__(self,root,transform=None):
        super(Data_pred, self).__init__()
        self.datas = glob(os.path.join(root,"*.jpg"))
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
        np.random.shuffle(self.paths)

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
        imgs = glob(os.path.join(path,"*"))
        # shuffle
        np.random.shuffle(imgs)
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

    def __getitem__(self,idx):
        path = self.datas[idx]
        img = PIL.Image.open(path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        # target
        target = self.classnames.index(os.path.basename(os.path.dirname(path)))

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target