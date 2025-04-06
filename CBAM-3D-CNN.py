from sklearn.decomposition import PCA
import torch.nn as nn
import torch.nn.functional as   F
import torch.optim as optim
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict
from torch.backends import cudnn
from operator import truediv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import time
import random
import torch
from tqdm import tqdm
import sys
import os
from scipy.io import loadmat

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)  # 7,3     3,1
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        out = x * self.ca(x)
        result = out * self.sa(out)
        return result


class TDCNN(nn.Module):
    def __init__(self, num_classes=3):
        super(TDCNN, self).__init__()
        self.layer1 = nn.Sequential(

            nn.Conv3d(1, 8, kernel_size=(7, 3, 3),
                      dilation=1, padding=(0, 1, 1), stride=(1, 2, 2)),

        nn.BatchNorm3d(8))
        self.cbam1 = CBAM(352)

        self.layer2 = nn.Sequential(
            nn.Conv3d(8, 16, kernel_size=3,
                      dilation=2, padding=(0, 1, 1), stride=(2, 2, 2)),

        nn.BatchNorm3d(16))
        self.cbam2 = CBAM(320)

        self.layer3 = nn.Sequential(
            nn.Conv3d(16, 32, kernel_size=3,
                      dilation=3, padding=(0, 1, 1), stride=(2, 2, 2)),

        nn.BatchNorm3d(32))
        self.cbam3 = CBAM(224)
        self.pooling = nn.MaxPool3d((1, 2, 2))

        # 其他卷积层
        self.conv8 = nn.Conv3d(32, 64, kernel_size=(1, 1, 1))
        self.conv16 = nn.Conv3d(64, 128, kernel_size=(1, 1, 1))
        self.conv32 = nn.Conv3d(128, 256, kernel_size=(1, 1, 1))
        self.conv32_res = nn.Conv3d(32, 256, kernel_size=(1, 1, 1))

        # self.conv32_sec = nn.Conv3d(256, 512, kernel_size=(1, 1, 1))
        # self.conv64_sec = nn.Conv3d(512, 512, kernel_size=(1, 1, 1))
        # self.conv128_sec = nn.Conv3d(512, 512, kernel_size=(1, 1, 1))
        # self.conv128_res = nn.Conv3d(256, 512, kernel_size=(1, 1, 1))

        # 全连接层
        self.linear1 = nn.Linear(1612800, 512)
        self.linear2 = nn.Linear(512, 128)
        self.linear3 = nn.Linear(128, num_classes)

    def forward(self, input):
        # 第一个卷积层和池化

        input = self.layer1(input)
        b, c, d, h, w = input.shape
        input = input.reshape(b, c*d, h, w)


        input = F.relu(self.cbam1(input)).reshape(b, c, d, h, w)

        # 第二个卷积层和池化
        input = self.layer2(input)
        b, c, d, h, w = input.shape
        input = input.reshape(b, c * d, h, w)
        input = F.relu(self.cbam2(input)).reshape(b, c, d, h, w)

        # 第三个卷积层和池化
        input = self.layer3(input)
        b, c, d, h, w = input.shape
        input = input.reshape(b, c * d, h, w)
        input = F.relu(self.cbam3(input)).reshape(b, c, d, h, w)

        # 其他卷积层和池化
        input1 = self.pooling(input)
        input = F.relu(self.conv8(input1))
        input = F.relu(self.conv16(input))

        input = self.conv32(input) + self.conv32_res(input1)
        # input = F.relu(input)
        # input = self.pooling(input)
        # input = self.conv128_sec(self.conv64_sec(self.conv32_sec(input))) + self.conv128_res(input)
        # 全局平均池化
        input = F.relu(input.squeeze(1))
        
        # 或者使用全局最大池化
        out = F.relu(self.linear1(input.flatten(1)))
        out = F.relu(self.linear2(out))
        out = self.linear3(out)
        # out = F.softmax(out, dim=1)
        return out



def applyPCA(X, numComponents):
    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))
    return newX


class MyDataset(Dataset):
    def __init__(self, file_path, numComponents=50):
        self.numComponents = numComponents
        self.data, labels, self.key = [], [], []

        for label in sorted(os.listdir(file_path)):
            for fname in os.listdir(os.path.join(file_path, label)):
                self.data.append(os.path.join(file_path, label, fname))
                labels.append(label)
                self.key.append(fname)
        self.label2index = {label: index for index, label in enumerate(sorted(set(labels)))}
        self.label_array = np.array([self.label2index[label] for label in labels])
        self.classes = self.label2index
        label_file = f'class_labels.txt'
        with open(label_file, 'w') as f:
            for id, label in enumerate(sorted(self.label2index)):
                f.writelines(str(id + 1) + ' ' + label + '\n')

    def __getitem__(self, index):  # 根据索引返回数据和对应的标签
        data = loadmat(self.data[index])[self.key[index].split(".")[0]]

        data=data[0:500,0:500,:]
        # print(self.label_array[index], self.key[index])
        data = self.applyPCA(data)
        data = torch.from_numpy(data).permute(2, 0, 1).unsqueeze(0).type(torch.FloatTensor)  # 转Float
        return data, torch.tensor(self.label_array[index])

    def applyPCA(self, X):
        newX = np.reshape(X, (-1, X.shape[2]))
        pca = PCA(n_components=self.numComponents, whiten=True)
        newX = pca.fit_transform(newX)
        newX = np.reshape(newX, (X.shape[0], X.shape[1], self.numComponents))
        return newX

    def __len__(self):  # 返回文件数据的数目
        return len(self.data)


def AA_andEachClassAccuracy(confusion_matrix):
    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc


def train_model(model, device, dataloader, optimizer, loss_fn, epoch, epochs):
    model.train()
    loss, accurate, sample_num = 0.0, 0.0, 0
    # 也可以for data, target in dataloader:       enumerate(dataloader,1)表示从1开始，不设置默认是0
    tqdm_dataloader = tqdm(dataloader, file=sys.stdout)
    for batch_index, (data, target) in enumerate(tqdm_dataloader):  # (data数据，target标签)都是tensor，target不是one_hot
        sample_num += data.shape[0]
        data,  target = data.to(device), target.long().to(device)
        optimizer.zero_grad()
        output = model(data)
        cur_loss = loss_fn(output, target)  # 交叉熵损失(多分类问题),是一个batchsize总的损失
        cur_loss.backward()
        pred = output.argmax(dim=1)  # 找到每行中数值最大的索引(其实就是这行的列） [0,0.1,0.2,0.3,0.4]返回为5表示数字5
        cur_acc = pred.eq(target.data.view_as(pred)).sum()  # 统计累积正确个数
        # 其中eq是判断是否相等，相等则计数，target.data.view_as(pred)为将target维度变成和pred一样的维度
        optimizer.step()  # 更新模型参数
        loss += cur_loss.item()
        accurate += cur_acc.item()
        tqdm_dataloader.desc = "[train epoch {}/{}] loss: {:.3f}, acc: {:.3f}, lr: {:.5f}".format(epoch + 1, epochs, loss / sample_num,
                                                                               accurate * 100 /sample_num, optimizer.param_groups[0]["lr"])  # 单个的损失
    train_acc = accurate / len(dataloader.dataset)  # len(dataloader.dataset)是数据总数，len(dataloader)是有几个batchsize
    loss /= len(dataloader.dataset)
    return loss, train_acc * 100


def val_model(model, device, dataloader, loss_fn, epoch, epochs):
    model.eval()
    loss, accurate, sample_num = 0.0, 0.0, 0
    with torch.no_grad():   # 测试不会计算梯度,也不会进行反向传播
        tqdm_dataloader = tqdm(dataloader, file=sys.stdout)
        for batch_index, (data,  target) in enumerate(tqdm_dataloader):   # for data, target in dataloader:是一样的
            sample_num += data.shape[0]
            data, target = data.to(device), target.long().to(device)
            output = model(data)
            cur_loss = loss_fn(output, target)
            pred = output.argmax(dim=1)    # 找到每行中数值最大的索引 [0,0.1,0.2,0.3,0.4]返回为5表示数字5
            cur_acc = pred.eq(target.data.view_as(pred)).sum()  # 统计累加正确个数
            loss += cur_loss.item()
            accurate += cur_acc.item()
            tqdm_dataloader.desc = "[val epoch {}/{}] loss: {:.3f}, acc: {:.3f}".format(epoch + 1, epochs, loss / sample_num,
                                                                                       accurate * 100 / sample_num)  # 单个的损失
        val_acc = accurate/len(dataloader.dataset)
        loss /= len(dataloader.dataset)
        return loss, val_acc * 100


def train_val_model(savepath, epochs, model, device, trainloader, valloader, optimizer, lossfn):
    min_acc, add_epoch = 0, 0
    # history = defaultdict(list)  # 构建一个默认value为list的字典
    train_start = time.time()

    # if opt.pre_train:
    #     weights_dict = torch.load(opt.pre_trainpath, map_location=device)
    #     for k in list(weights_dict.keys()):
    #         if "head" in k:
    #             del weights_dict[k]
    #     model.load_state_dict(weights_dict, strict=False)
    #     print(model.load_state_dict(weights_dict, strict=False))
    #     min_acc = opt.val_acc
    #     add_epoch = opt.add_epoch

    for epoch in range(epochs - add_epoch):
        epoch = epoch + add_epoch
        train_loss, train_accuracy = train_model(model, device, trainloader, optimizer, lossfn, epoch, epochs)
        #scheduler.step() # 更新学习率
        val_loss, val_accuracy = val_model(model, device, valloader, lossfn, epoch, epochs)

        with open(train_message, 'a') as t_f:  # 将结果写在文件里
            t_f.write(f"epoch{epoch + 1}, train_loss:{train_loss},train_acc:{train_accuracy}, val_acc:{val_accuracy},val_loss:{val_loss}" + "\n")

        # history['train_acc'].append(train_accuracy)
        # history['train_loss'].append(train_loss)
        # history['val_acc'].append(val_accuracy)
        # history['val_loss'].append(val_loss)

        writer.add_scalars("Acc", {"train": train_accuracy, "val": val_accuracy}, epoch)
        writer.add_scalars("Loss", {"train": train_loss, "val": val_loss}, epoch)

        writer.add_scalar("train_acc", train_accuracy, epoch)
        writer.add_scalar("train_loss", train_loss, epoch)
        writer.add_scalar("val_acc", val_accuracy, epoch)
        writer.add_scalar("val_loss", val_loss, epoch)
        # 保存最好的模型
        if val_accuracy > min_acc:
            if not os.path.exists(savepath):
                os.mkdir(savepath)
            min_acc = val_accuracy

            torch.save(model.state_dict(), f"{savepath}/best_model_weight.pth")
            with open(train_message, 'w') as t_f:  # 将结果写在文件里
                t_f.write(f"第{epoch + 1}轮,best_val_acc:{val_accuracy}" + "\n")
            print(f"save best model, 第{epoch + 1}轮")

        # # 每隔20轮保存
        # if (epoch + 1) % 20 == 0:
        #     torch.save(model.state_dict(), f"{savepath}/{epoch + 1}.pth")
        # 保存最后一轮
        if (epoch + 1) == epochs:
            torch.save(model.state_dict(), f"{savepath}/last_model_weight.pth")

    train_end = time.time()
    train_time = train_end - train_start
    time_train_now = time.strftime("%Y-%m-%d %H:%M:%S")

    with open(train_message, 'a') as t_f:  # 将结果写在文件里
        t_f.write(f"现在时间:{time_train_now}" + "\n")
        t_f.write(f"训练{epochs}轮用时{train_time/60}分钟！" + "\n")
    print(f"训练{epochs}轮用时{train_time/60}分钟！")
    writer.close()

def test_model(savepath, model, device, dataloader, loss_fn):
    model.eval()
    class_name = dataloader.dataset.classes
    count, loss, accurate = 0, 0.0, 0.0
    test_start = time.time()
    with torch.no_grad():   # 测试不会计算梯度,也不会进行反向传播
        tqdm_dataloader = tqdm(dataloader, file=sys.stdout)
        for batch_index, (data, target) in enumerate(tqdm_dataloader):   # for data, target in dataloader:是一样的
            data, target = data.to(device), target.long().to(device)
            output = model(data)
            cur_loss = loss_fn(output, target)
            pred = output.argmax(dim=1)     # 找到每行中数值最大的索引 [0,0.1,0.2,0.3,0.4]返回为5表示数字5
            cur_acc = pred.eq(target.data.view_as(pred)).sum()  # 统计累加正确个数
            # 其中eq是判断是否相等，相等则计数sum进行累加，target.data.view_as(pred)为将target维度变成和pred一样的维度

            outputs = np.argmax(output.detach().cpu().numpy(), axis=1)
            target = target.cpu()
            if count == 0:
                y_pred = outputs
                y_test = target
                count = 1
            else:
                y_pred = np.concatenate((y_pred, outputs))  # np.concatenate()是用来对数列或矩阵进行合并的
                y_test = np.concatenate((y_test, target))

            loss += cur_loss.item()
            accurate += cur_acc.item()
        test_acc = accurate/len(dataloader.dataset)
        loss /= len(dataloader.dataset)
        print(f"test_loss : {loss}")
        print(f"test_acc : {100 * test_acc}")

        classification = classification_report(y_test, y_pred, target_names=class_name)
        oa = accuracy_score(y_test, y_pred)
        confusion = confusion_matrix(y_test, y_pred)  # 计算TP, FP, TN, FN
        df_cm = pd.DataFrame(confusion, index=class_name, columns=class_name)
        plt.figure(figsize=(10, 10))
        sns.heatmap(df_cm, fmt="d", annot=True, cmap="Oranges")
        plt.ylabel('True Label')
        plt.xlabel('Pred Label')
        plt.savefig(f"{savepath}/confusion_{epochs}.png")
        each_acc, aa = AA_andEachClassAccuracy(confusion)
        kappa = cohen_kappa_score(y_test, y_pred)
        classification = str(classification)
        confusion = str(confusion)

        test_end = time.time()
        test_time = test_end - test_start
        time_test_now = time.strftime("%Y-%m-%d %H:%M:%S")

        with open(train_message, 'a') as t_f:  # 将结果写在文件里
            t_f.write(f"现在时间:{time_test_now}" + "\n")
            t_f.write(f"测试时间:{test_time}秒！"+"\n")
            t_f.write(f'Test loss:{loss}' + "\n")
            t_f.write(f'Test accuracy:{test_acc * 100}' + "\n")
            t_f.write(f'Kappa accuracy:{kappa * 100}' + "\n")
            t_f.write(f'Each accuracy:{each_acc * 100}' + "\n")
            t_f.write(f'Overall accuracy:{oa * 100}' + "\n")
            t_f.write(f'Average accuracy:{aa * 100}' + "\n")
            t_f.write(f'{classification}' + "\n")
            t_f.write(f'{confusion}' + "\n")


if __name__ == '__main__':
    #t_num = random.sample(range(0, 200), 40)
    #test_num=t_num[0:20]
    '''
    print(len(test_num))
    print(test_num)
    val_num = t_num [20:40]
    print(len(val_num))
    print(val_num)
    train_num = []
    j = 0
    z = 0
    test_num.sort()
    val_num.sort()
    for i in range(0, 200):
        if i != test_num[j] and i != val_num[z]:
            train_num.append(i)
        if i == test_num[j]:
            j = j + 1
            if j == 20:
                j = 19

        if i == val_num[z]:
            z = z + 1
            if z == 20:
                z = 19
    '''
    seed = 0
    random.seed(seed)  # 保证shuffle每次打乱后结果一样
    np.random.seed(seed)
    torch.manual_seed(seed)  # torch+CPU
    torch.cuda.manual_seed(seed)  # torch+GPU
    class_names = ['symptomatic', "health", "mildew"]
    nc = len(class_names)
    save_path = "results"
    os.makedirs(save_path, exist_ok=True)  # 创建文件夹
    train_message = "results/reault.txt"
    trainset = MyDataset("F:/Data/Data500×500_Mat/train")
    # testset = MyDataset("F:/Data/Data500×500_Mat/test")
    # valset = MyDataset("F:/Data/Data500×500_Mat/val")
    testset = MyDataset("F:/Data/Data750×700_Mat/Ttest")
    valset = MyDataset("F:/Data/Data750×700_Mat/Vval")

    val_loader = DataLoader(dataset=valset, batch_size=4, shuffle=True, num_workers=2)
    test_loader = DataLoader(dataset=testset,  batch_size=4, shuffle=True, num_workers=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TDCNN(num_classes=nc).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-6)  # optim.Adam
    epochs =15
    cudnn.benchmark = True
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    writer = SummaryWriter(f"{save_path}/logs")
    writer.close()
    writer.add_graph(model, torch.randn(1, 1, 50, 500, 500).cuda())

    train_loader = DataLoader(dataset=trainset, batch_size=4, shuffle=True, num_workers=0)
    train_val_model(save_path, epochs, model, device, train_loader, val_loader, optimizer, criterion)  # 训练并验证
    model_weight_path = f"{save_path}/best_model_weight.pth"
    print("保存")
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    test_model(save_path, model, device, test_loader, criterion)