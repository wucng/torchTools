import torch
import torch.nn.functional as F
import torch.nn as nn

class LossFunc(object):
    """
    自定义loss 函数
    """
    def __init__(self,num_classes=10,reduction="mean",esp = 1e-5):
        self.num_classes = num_classes
        self.reduction = reduction
        self.esp = esp

    def cross_entropy00(self,input,target): # 0.98
        """输入的input不经过softmax
        pred = [32.8,1.2,3.4,-9.8,-5.8]
        true = [0,1,0,0,0]
        pred = exp(pred)
        loss = -log(true*pred/sum(pred)) # 让 pred对应的第2列越大越好 pred/sum(pred) 其实等价于使用了softmax
        """
        # 转one-hot
        if target.ndim==1:
            target = torch.eye(self.num_classes, self.num_classes,device=target.device)[target]
            # target = F.one_hot(target,num_classes=self.num_classes,device=target.device)
        input = torch.exp(input)
        loss = -torch.log(target*input/(torch.sum(input,dim=1,keepdim=True))+self.esp)
        if self.reduction=="sum":
            loss = torch.sum(loss)
        elif self.reduction=="mean":
            loss = torch.mean(loss)
        else:
            raise("reduction must in [sum,mean]")

        return loss

    def cross_entropy01(self,input,target): # 0.99
        """输入的input不经过softmax
        pred = [32.8,1.2,3.4,-9.8,-5.8]
        true = [0,1,0,0,0]
        pred = exp(pred)
        loss = -true*log(pred/sum(pred))
        """
        # 转one-hot
        if target.ndim==1:
            target = torch.eye(self.num_classes, self.num_classes,device=target.device)[target]
        input = torch.exp(input)
        loss = -target * torch.log(input/(torch.sum(input,dim=1,keepdim=True))+self.esp)
        if self.reduction=="sum":
            loss = torch.sum(loss)
        elif self.reduction=="mean":
            loss = torch.mean(loss)
        else:
            raise("reduction must in [sum,mean]")

        return loss

    def cross_entropy02(self,input,target): # 0.99
        """输入的input不经过softmax
        pred = [32.8,1.2,3.4,-9.8,-5.8]
        pred = softmax(pred)
        true = [0,1,0,0,0]
        loss = -log(true*pred) # 让 pred对应的第2列越大越好
        """
        # 转one-hot
        if target.ndim==1:
            target = torch.eye(self.num_classes, self.num_classes,device=target.device)[target]
        input = torch.softmax(input,-1)
        if self.reduction=="sum":
            # loss = torch.sum(-torch.log(input[target])) #  target 非one-hot
            loss = torch.sum(-torch.log(torch.sum(target*input,1)))
        elif self.reduction=="mean":
            # loss = torch.mean(-torch.log(input[target]))
            loss = torch.mean(-torch.log(torch.sum(target * input, 1)))
        else:
            raise("reduction must in [sum,mean]")

        return loss

    def cross_entropy03(self,input,target): # 0.99
        """输入的input不经过softmax
        pred = [32.8,1.2,3.4,-9.8,-5.8]
        pred = softmax(pred)
        true = [0,1,0,0,0]
        loss = -true*log(pred) # 让 pred对应的第2列越大越好

        # or
        pred = [32.8,1.2,3.4,-9.8,-5.8]
        true = [0,1,0,0,0]
        loss = -true*log_softmax(pred) # log_softmax(x) 等价于 log(softmax(x))
        """
        # 转one-hot
        if target.ndim==1:
            target = torch.eye(self.num_classes, self.num_classes,device=target.device)[target]
        # input = torch.softmax(input,-1)
        loss = -target*torch.log_softmax(input,-1)
        if self.reduction=="sum":
            loss = torch.sum(loss)
        elif self.reduction=="mean":
            loss = torch.mean(loss)
        else:
            raise("reduction must in [sum,mean]")

        return loss

    def binary_cross_entropy(self,input,target,use_sigmoid=True):
        """
        loss = -(true*log(pred)+(1-true)*log(1-pred)) # pred 使用sigmoid处理
        :param input: [n,]
        :param target: [n,]
        :return:
        """
        if use_sigmoid:
            input = torch.sigmoid(input)

        loss = -1 * (target * torch.log(input) + (1 - target) * torch.log(1 - input))

        if self.reduction == "sum":
            loss = torch.sum(loss)
        elif self.reduction == "mean":
            loss = torch.mean(loss)
        else:
            raise ("reduction must in [sum,mean]")

        return loss

    def multilabel2MulTwoLabel(self,input,target): # 0.99
        """输入的input不经过softmax
        # 每列看成一个二分类，拆成多个2分类
        pred = [32.8,1.2,3.4,-9.8,-5.8]
        pred = softmax(pred)
        true = [0,1,0,0,0]

        loss = 0
        for i in range(5):
            loss += -(true[i]*log(pred[i])+(1-true[i])*log(1-pred[i]))
        """
        # 转one-hot
        if target.ndim == 1:
            target = torch.eye(self.num_classes, self.num_classes,device=target.device)[target]
        input = torch.softmax(input,-1)

        """
        loss = 0
        for i in range(self.num_classes):
            # loss+=self.binary_cross_entropy(input[:,i],target[:,i],use_sigmoid=False) # 使用了softmax，相当于已经用过sigmoid
            loss+=self.focal_binary_cross_entropy(input[:,i],target[:,i],use_sigmoid=False) # 使用了softmax，相当于已经用过sigmoid
        """
        # loss = self.binary_cross_entropy(input.view(-1),target.view(-1),use_sigmoid=False) # [n,10] ->[n*10,]
        loss = self.focal_binary_cross_entropy(input.view(-1),target.view(-1),use_sigmoid=False)
        # """

        return loss

    def hinge_loss(self,input,target): # 0.99
        """input不使用softmax
        SVM 分类loss,也可以用于二分类 (不推荐)

        pred = [32.8,1.2,3.4,-9.8,-5.8]
        true = [0,1,0,0,0]

        index = np.argmax(ture) # 2

        loss = 0
        for i in range(5):
            if i != index:
                loss += max(0,pred[i]-pred[index]+1)

        :param input [n,10]
        :param target [n,] 非one-hot
        """
        # 转one-hot
        if target.ndim == 1:
            target = torch.eye(self.num_classes, self.num_classes,device=target.device)[target]

        # input = torch.softmax(input, -1)

        loss = torch.max(torch.max(torch.zeros_like(input), input - torch.sum(input * target, -1, keepdim=True) + 1),-1)[0]
        if self.reduction=="sum":
            loss = torch.sum(loss)
        elif self.reduction=="mean":
            loss = torch.mean(loss)
        else:
            raise("reduction must in [sum,mean]")
        return loss

    def mse(self,input,target): # 0.98
        """
        loss = sum((pred-ture)**2)/(2m)
        :param input:
        :param target:
        :return:
        """
        # 转one-hot
        if target.ndim == 1:
            target = torch.eye(self.num_classes, self.num_classes,device=target.device)[target]
        input = torch.softmax(input,-1)
        loss = (input-target)**2
        if self.reduction=="sum":
            loss = torch.sum(loss)
        elif self.reduction=="mean":
            loss = torch.mean(loss)
        else:
            raise("reduction must in [sum,mean]")
        return loss

    def l1_loss(self,input,target): # 0.97
        """
        loss = sum(abs(pred-true)/(2m)
        :param input:
        :param target:
        :return:
        """
        # 转one-hot
        if target.ndim == 1:
            target = torch.eye(self.num_classes, self.num_classes,device=target.device)[target]
        input = torch.softmax(input,-1)
        loss = torch.abs(input-target)
        if self.reduction=="sum":
            loss = torch.sum(loss)
        elif self.reduction=="mean":
            loss = torch.mean(loss)
        else:
            raise("reduction must in [sum,mean]")
        return loss

    def smooth_l1_loss(self,input,target,alpha=0.5): # 0.98
        """
        tmp = pred - true
        if abs(tmp)<1:
            loss = 0.5*tmp**2
        else:
            loss = abs(tmp)-0.5
        :param input:
        :param target:
        :return:
        """
        # 转one-hot
        if target.ndim == 1:
            target = torch.eye(self.num_classes, self.num_classes,device=target.device)[target]
        input = torch.softmax(input,-1)
        tmp = input-target
        abs_tmp = torch.abs(tmp)
        loss = alpha*tmp**2*(abs_tmp<1).float()+ (abs_tmp-0.5)*(abs_tmp>=1).float()

        if self.reduction=="sum":
            loss = torch.sum(loss)
        elif self.reduction=="mean":
            loss = torch.mean(loss)
        else:
            raise("reduction must in [sum,mean]")
        return loss

    def log_loss(self,input,target): # 0.82
        """
        # MSE
        loss = sum((pred-ture)**2)/(2m)
        # -->
        loss = sum(log((pred-ture)**2))/(2m)
        # -->
        loss = sum(log(pred-true))/m # 前提 pred-true>0

        :param input:
        :param target:
        :return:
        """
        # 转one-hot
        if target.ndim == 1:
            target = torch.eye(self.num_classes, self.num_classes,device=target.device)[target]

        # loss = torch.log_softmax(input-target,-1)
        loss = torch.exp(abs(input-target))

        if self.reduction=="sum":
            loss = torch.sum(loss)
        elif self.reduction=="mean":
            loss = torch.mean(loss)
        else:
            raise("reduction must in [sum,mean]")
        return loss

    def focal_binary_cross_entropy(self,input,target,alpha=0.25,gamma=2,use_sigmoid=True):
        """
        pred = sigmoid(pred)
        floss = -(alpha*(1-pred)^gamma*true*log(pred)+(1-alpha)*pred^gamma*(1-true)*log(1-pred)  # 对应pred为1列
        alpha 为正样本权重，1-alpha 为负样本权重
        gamma 控制样本分类的难易程度，越容易的权重小，越难的权重大

        :param input:
        :param target:
        :param alpha:
        :param gamma:
        :return:
        """

        if use_sigmoid:
            input = torch.sigmoid(input)

        # BCE_loss = -1*(target * torch.log(input) + (1 - target) * torch.log(1 - input))
        BCE_loss = F.binary_cross_entropy(input,target,reduction="sum")
        pt = torch.exp(-BCE_loss)
        loss = alpha * (1 - pt) ** gamma * BCE_loss

        # loss = -1*(alpha*(1-input)**gamma*target * torch.log(input) + (1-alpha)*input**gamma*(1 - target) * torch.log(1 - input))

        if self.reduction=="sum":
            loss = torch.sum(loss)
        elif self.reduction=="mean":
            loss = torch.mean(loss)
        else:
            raise("reduction must in [sum,mean]")

        return loss

    def focal_cross_entropy(self,input,target,alpha=None,gamma=2): # 0.99
        """输入的input不经过softmax
        floss = -alpha*(1-true*pred)^gamma*log(true*pred)
        floss = -alpha*(1-pred)^gamma*true*log(pred)

        alpha 控制每个类别的权重
        gamma 控制样本分类的难易程度，越容易的权重小，越难的权重大

        """
        # 转one-hot
        if target.ndim==1:
            target = torch.eye(self.num_classes, self.num_classes,device=target.device)[target]

        # loss = -target*torch.log_softmax(input,-1)
        if alpha is None:
            alpha = torch.ones(self.num_classes,dtype=target.dtype,device=target.device)
        else: # alpha = [1,0.5,...], len(alpha) = num_classes
            alpha = torch.as_tensor(alpha, dtype=target.dtype, device=target.device)
        # input = torch.softmax(input, -1)
        # loss = -alpha * (1 - input) ** gamma * target * torch.log(input)
        # loss = -alpha*(1-torch.sum(target*input,-1))**gamma*torch.log(torch.sum(target*input,-1))

        # BCE_loss = -target*torch.log_softmax(input,-1)#-0.2*torch.softmax(input, -1)*torch.log_softmax(input,-1)
        BCE_loss = F.cross_entropy(input, targets, reduction="sum")
        pt = torch.exp(-BCE_loss)
        loss = alpha * (1 - pt) ** gamma * BCE_loss



        if self.reduction=="sum":
            loss = torch.sum(loss)
        elif self.reduction=="mean":
            loss = torch.mean(loss)
        else:
            raise("reduction must in [sum,mean]")

        return loss

    def focal_mse(self,input,target,alpha=None,gamma=2): # 0.99
        """
        pred = softmax(pred)
        loss = sum(alpha*(1-pred)**gamma*(pred-ture)**2)/(2m)

        alpha 控制每个类别的权重
        gamma 控制样本分类的难易程度，越容易的权重小，越难的权重大

        :param input:
        :param target:
        :return:
        """
        # 转one-hot
        if target.ndim == 1:
            target = torch.eye(self.num_classes, self.num_classes,device=target.device)[target]
        input = torch.softmax(input,-1)
        # loss = (input-target)**2
        if alpha is None:
            alpha = torch.ones(self.num_classes,dtype=target.dtype,device=target.device)
        else: # alpha = [1,0.5,...], len(alpha) = num_classes
            alpha = torch.as_tensor(alpha, dtype=target.dtype, device=target.device)

        # loss = alpha*(1-input)**gamma*(input-target)**2
        # BCE_loss = (input-target)**2
        BCE_loss = F.mse_loss(input,target,reduction="sum")
        pt = torch.exp(-BCE_loss)
        loss = alpha * (1 - pt) ** gamma * BCE_loss


        if self.reduction=="sum":
            loss = torch.sum(loss)
        elif self.reduction=="mean":
            loss = torch.mean(loss)
        else:
            raise("reduction must in [sum,mean]")
        return loss