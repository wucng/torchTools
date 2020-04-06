from __future__ import print_function
import os
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils import clip_grad_norm_
from torchvision import datasets, transforms
import torchvision
# from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter
import PIL.Image
from collections import OrderedDict
import math
import matplotlib.pyplot as plt
import json

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
    def forward(self, x):
        return torch.flatten(x,1)

def get_model(num_classes,droprate):
    _model = torchvision.models.resnet50(True)
    network = nn.Sequential(
        _model.conv1,
        _model.bn1,
        _model.relu,
        _model.maxpool,
        _model.layer1,
        _model.layer2,
        _model.layer3,
        _model.layer4,

        nn.Dropout(droprate),
        nn.AdaptiveAvgPool2d((1, 1)),
        Flatten(),
        nn.Linear(2048, num_classes)
    )
    return network


class History():
    epoch = []
    history = {"loss": [], "acc": [], "val_loss": [], "val_acc": []}

    # 打印训练结果信息
    def show_final_history(self,history):
        fig, ax = plt.subplots(1, 2, figsize=(15, 5))
        ax[0].set_title('loss')
        ax[0].plot(history.epoch, history.history["loss"], label="Train loss")
        ax[0].plot(history.epoch, history.history["val_loss"], label="Validation loss")
        ax[1].set_title('acc')
        ax[1].plot(history.epoch, history.history["acc"], label="Train acc")
        ax[1].plot(history.epoch, history.history["val_acc"], label="Validation acc")
        ax[0].legend()
        ax[1].legend()

        plt.show()



class ClassifyModel(nn.Module):
    def __init__(self,num_classes,epochs=10,droprate=0.5,lr=1e-3,
                 batch_size=32,test_batch_size=64,log_interval=30,
                 train_dataset=None,test_dataset=None,pred_dataset=None,
                 network=None, optimizer=None,lossFunc=None,
                 base_path="./",save_model="model.pt",
                 useTensorboard=False,useAdvance=False,useParallel=False,parallels=[0]):
        """parallels:GPU编号"""
        super(ClassifyModel,self).__init__()
        self.num_classes = num_classes
        self.epochs = epochs
        self.droprate = droprate
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.log_interval = log_interval
        seed = 100
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        torch.manual_seed(seed)
        kwargs = {'num_workers': 5, 'pin_memory': True} if self.use_cuda else {}
        self.base_path = base_path

        if train_dataset is not None:
            self.train_loader = DataLoader(train_dataset,batch_size=self.batch_size,shuffle=True,**kwargs)
        if test_dataset is not None:
            self.test_loader = DataLoader(test_dataset,batch_size=self.test_batch_size,shuffle=False,**kwargs)
        if pred_dataset is not None:
            self.pred_loader = DataLoader(pred_dataset,batch_size=self.test_batch_size,shuffle=False,**kwargs)

        if network is None:
            self.network = get_model(num_classes,droprate)
        else:
            self.network = network

        if self.use_cuda:
            self.network.to(self.device)

        if lossFunc is None:
            self.lossFunc = nn.CrossEntropyLoss(reduction='sum')
        else:
            self.lossFunc = lossFunc

        if optimizer is None:
            self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr, weight_decay=4e-05)
        else:
            # if hasattr(self.network, 'params'):
            if callable(self.network.params):
                print("选择自定义参数")
                self.optimizer = optimizer(self.network.params(lr), lr=lr, weight_decay=4e-5)
            else:
                self.optimizer = optimizer(self.network.parameters(),lr=lr,weight_decay=4e-5)

        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=0.1)

        self.history = History()

        # self.save_model = os.path.join(os.getcwd(),save_model)
        self.save_model = os.path.join(base_path,save_model)
        if os.path.exists(self.save_model):
            # model.load_state_dict(torch.load(self.save_model))
            state_dict = torch.load(self.save_model)
            self.network.load_state_dict({k: v for k, v in state_dict.items() if k in self.network.state_dict()})

        self.useTensorboard = useTensorboard
        if useTensorboard:
            self.writer = SummaryWriter(os.path.join(base_path,'runs/experiment'))
        self.useParallel = useParallel
        self.useAdvance = useAdvance

        if self.useParallel:
            self.network = nn.DataParallel(self.network)
            os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, parallels))


    # Train the network for one or more epochs, validating after each epoch.
    def train(self, epoch=0):
        self.network.train()
        train_loss = 0
        correct = 0  # 正确的个数
        num_trains = len(self.train_loader.dataset)
        for batch, (data, target) in enumerate(self.train_loader):
            if self.useTensorboard:
                # create grid of images
                img_grid = torchvision.utils.make_grid(data)
                # write to tensorboard
                self.writer.add_image('four_mnist_images', img_grid)

            # data, target = Variable(data), Variable(target)
            if self.use_cuda:
                data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.network(data)
            loss = self.lossFunc(output, target)
            train_loss += loss.item()
            loss /= len(data)
            loss.backward()
            self.optimizer.step()

            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

            if batch % self.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch * len(data),
                                                                               len(self.train_loader.dataset),
                                                                               100. * batch / len(
                                                                                   self.train_loader),
                                                                               loss.data.item()))

                if self.useTensorboard:
                    # ...log the running loss
                    self.writer.add_scalar('training loss',
                                           loss,
                                           (epoch - 1) * num_trains + batch)

        train_loss /= num_trains
        train_acc = correct / num_trains

        print('Train, Average Loss: {:.6f}\t,acc:{:.6f}'.format(
            train_loss, train_acc))

        return train_loss, train_acc

    def accuracy(self, output, target):
        _, argmax = torch.max(output, 1)
        accuracy = (target == argmax.squeeze()).float().mean()
        return accuracy

    def __train(self, epoch=0): # 强化版
        self.network.train()
        mixup_alpha = 1.0
        ricap_beta = 0.3
        train_loss = 0
        accs = []
        num_trains = len(self.train_loader.dataset)

        # 设置一个随机数，来选择增强方式
        # 1.普通方式
        # 2.ricap
        # 3.mixup

        for batch_idx, (data, target) in enumerate(self.train_loader):
            state = np.random.choice(["general", "ricap", "mixup"], 1)[0]

            if state == "general":
                data, target = data.to(self.device), target.to(self.device)
                output = self.network(data)
                loss = self.lossFunc(output, target)
                acc = self.accuracy(output, target)

            elif state == "ricap":
                # ricap 数据随机裁剪组合增强
                I_x, I_y = data.size()[2:]

                w = int(np.round(I_x * np.random.beta(ricap_beta, ricap_beta)))
                h = int(np.round(I_y * np.random.beta(ricap_beta, ricap_beta)))
                w_ = [w, I_x - w, w, I_x - w]
                h_ = [h, h, I_y - h, I_y - h]

                cropped_images = {}
                c_ = {}
                W_ = {}
                for k in range(4):
                    idx = torch.randperm(data.size(0))
                    x_k = np.random.randint(0, I_x - w_[k] + 1)
                    y_k = np.random.randint(0, I_y - h_[k] + 1)
                    cropped_images[k] = data[idx][:, :, x_k:x_k + w_[k], y_k:y_k + h_[k]]
                    # c_[k] = target[idx].cuda()
                    c_[k] = target[idx].to(self.device)
                    W_[k] = w_[k] * h_[k] / (I_x * I_y)

                patched_images = torch.cat(
                    (torch.cat((cropped_images[0], cropped_images[1]), 2),
                     torch.cat((cropped_images[2], cropped_images[3]), 2)),
                    3)
                # patched_images = patched_images.cuda()
                patched_images = patched_images.to(self.device)
                output = self.network(patched_images)

                # loss = sum([W_[k] * criterion(output, c_[k]) for k in range(4)])
                loss = sum([W_[k] * self.lossFunc(output, c_[k]) for k in range(4)])

                # acc = sum([W_[k] * accuracy(output, c_[k])[0] for k in range(4)])
                acc = sum([W_[k] * self.accuracy(output, c_[k]) for k in range(4)])

            else:  # mixup
                l = np.random.beta(mixup_alpha, mixup_alpha)
                idx = torch.randperm(data.size(0))
                input_a, input_b = data, data[idx]
                target_a, target_b = target, target[idx]

                mixed_input = l * input_a + (1 - l) * input_b

                target_a = target_a.to(self.device)
                target_b = target_b.to(self.device)
                mixed_input = mixed_input.to(self.device)
                output = self.network(mixed_input)
                # loss = l * criterion(output, target_a) + (1 - l) * criterion(output, target_b)
                loss = l * self.lossFunc(output, target_a) + (1 - l) * self.lossFunc(output, target_b)
                # acc = l * accuracy(output, target_a)[0] + (1 - l) * accuracy(output, target_b)[0]
                acc = l * self.accuracy(output, target_a) + (1 - l) * self.accuracy(output, target_b)

            train_loss += loss.item()
            loss /= len(data)
            self.optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), 1.0 - 1e-10)  # 梯度裁剪
            self.optimizer.step()

            accs.append(acc.item())

            if batch_idx % 20 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), num_trains,
                           100. * batch_idx * len(data) / num_trains, loss.item()))

        train_loss /= num_trains
        train_acc = np.mean(accs)

        print('Train, Average Loss: {:.6f}\t,acc:{:.6f}'.format(
            train_loss, train_acc))

        return train_acc, train_loss

    # Test the network
    def test(self, epoch=0):
        self.network.eval()
        test_loss = 0
        correct = 0
        num_tests = len(self.test_loader.dataset)
        for data, target in self.test_loader:
            with torch.no_grad():
                # data, target = Variable(data), Variable(target)
                if self.use_cuda:
                    data, target = data.to(self.device), target.to(self.device)

            output = self.network(data)
            test_loss += self.lossFunc(output, target).data.item()
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= num_tests
        test_acc = correct / num_tests
        print(
            '\nTest Epoch: {} Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(epoch, test_loss,
                                                                                                  correct,
                                                                                                  num_tests,
                                                                                                  100. * test_acc))

        return test_loss, test_acc

    def fit(self):
        for e in range(self.epochs):
            if self.useAdvance:
                train_loss, train_acc = self.__train(e + 1)
            else:
                train_loss, train_acc = self.train(e + 1)
            test_loss, test_acc = self.test(e + 1)
            torch.save(self.network.state_dict(), self.save_model)  # save models

            # update the learning rate
            self.lr_scheduler.step()

            # 记录每个epoch的 loss accuracy
            self.history.epoch.append(e)
            self.history.history["loss"].append(train_loss)
            self.history.history["acc"].append(train_acc)
            self.history.history["val_loss"].append(test_loss)
            self.history.history["val_acc"].append(test_acc)

        # 保存json文件
        json.dump(self.history.history, open(os.path.join(self.base_path,"result.json"), "w"))
        # 显示训练记录的结果
        self.history.show_final_history(self.history)

    def predict(self):
        self.network.eval()
        imgsPath = []
        labels = []
        with torch.no_grad():
            for data, path in self.pred_loader:
                if self.use_cuda:
                    data = data.to(self.device)
                output = self.network(data)
                pred = output.max(1, keepdim=True)[1]
                imgsPath.extend(path)
                labels.extend(pred.detach().cpu().numpy())

        return imgsPath, labels

    def getTrueAndPred(self,wantFeature=False):
        self.network.eval()
        y_true = []
        y_pred = []
        feature = []
        with torch.no_grad():
            for data, target in self.test_loader:
                if self.use_cuda:
                    data, target = data.to(self.device), target.to(self.device)
                output = self.network(data)
                pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
                y_true.extend(target.to("cpu").numpy())
                y_pred.extend(pred.to("cpu").numpy())
                if wantFeature:
                    feature.extend(output.to("cpu").numpy())
        return y_true, y_pred,feature