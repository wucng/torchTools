from __future__ import print_function
import os
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision
import numpy as np
from torch.utils.data import DataLoader, Dataset
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



class ClassifyModel(nn.Module):
    def __init__(self,num_classes,epochs=10,droprate=0.5,lr=1e-3,
                 batch_size=32,test_batch_size=64,log_interval=30,
                 train_dataset=None,test_dataset=None,pred_dataset=None,
                 network=None, optimizer=None,lossFunc=None,history=None,
                 base_path="./",save_model="model.pt",
                 useTensorboard=False):
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
            # if hasattr(self.network, 'parmas'):
            if callable(self.network.params):
                print("选择自定义参数")
                self.optimizer = optimizer(self.network.parmas(lr), lr=lr, weight_decay=4e-5)
            else:
                self.optimizer = optimizer(self.network.parameters(),lr=lr,weight_decay=4e-5)

        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=0.1)

        self.history = history
        # self.save_model = os.path.join(os.getcwd(),save_model)
        self.save_model = os.path.join(base_path,save_model)
        if os.path.exists(self.save_model):
            # model.load_state_dict(torch.load(self.save_model))
            state_dict = torch.load(self.save_model)
            self.network.load_state_dict({k: v for k, v in state_dict.items() if k in self.network.state_dict()})

        self.useTensorboard = useTensorboard
        if useTensorboard:
            self.writer = SummaryWriter(os.path.join(base_path,'runs/experiment'))

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
            pred = output.data.max(1)[1]
            correct += pred.eq(target.data).cpu().sum()
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