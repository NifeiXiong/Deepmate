import pandas as pd
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from sklearn.preprocessing import LabelEncoder
import scipy.io as io
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch
from torch.optim import Adam
import numpy as np
from utils import metrics, cost, safeCreateDir
import time
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

labels = []  # 存放标签


# 46读入 302,309,325行输出以及输出读入

def get_labels_inter1(df1):  # 获得标签
    global labels

    b = df1['y']
    n = np.sum(b > 0)
    n1 = df1["interval"].median()
    # print(b)
    if n > 0.4 * b.size:
        x = 1
    else:
        x = 0
    labels.append(x)
    return n1


def getFormatData(pre_read_excel_address):
    global labels
    # sheet = pd.read_excel(io=r'..\USV_data\input\usv_test.xlsx')
    sheet = pd.read_excel(io=pre_read_excel_address)
    sheet.head()
    df = pd.DataFrame(
        columns=['Score', 'Call Length (s)', 'Principal Frequency (kHz)', 'Low Freq (kHz)', 'High Freq (kHz)',
                 'Delta Freq (kHz)', 'Frequency Standard Deviation (kHz)', 'Slope (kHz/s)', 'Sinuosity',
                 'Mean Power (dB/Hz)', 'Tonality', 'Peak Freq (kHz)', 'interval', 'y'])  # 记录的表
    df1 = pd.DataFrame(
        columns=['Score', 'Call Length (s)', 'Principal Frequency (kHz)', 'Low Freq (kHz)', 'High Freq (kHz)',
                 'Delta Freq (kHz)', 'Frequency Standard Deviation (kHz)', 'Slope (kHz/s)', 'Sinuosity',
                 'Mean Power (dB/Hz)', 'Tonality', 'Peak Freq (kHz)', 'interval', 'y'])  # 记录的表
    last = pd.DataFrame(
        columns=['Score', 'Call Length (s)', 'Principal Frequency (kHz)', 'Low Freq (kHz)', 'High Freq (kHz)',
                 'Delta Freq (kHz)', 'Frequency Standard Deviation (kHz)', 'Slope (kHz/s)', 'Sinuosity',
                 'Mean Power (dB/Hz)', 'Tonality', 'Peak Freq (kHz)', 'interval', 'n', 'y'])  # 最终所求表
    print(sheet.head())
    a = sheet['interval']
    fin = []
    print(a)
    for i in range(len(a)):
        df.loc[0] = sheet.loc[i]
        #        print(i)
        if i == 0:
            # df1 = last.concat(df, ignore_index=True)
            df1 = pd.concat([df1, df])
        else:
            if a[i] > 1.5:
                n1 = get_labels_inter1(df1)  # 计算的函数df1--->df2
                df1.interval[df1.interval > 1.5] = n1
                n = len(df1)
                df1['n'] = df.apply(lambda x: n, axis=1)
                df1.pop('y')
                #                print(df1)
                df2 = torch.Tensor(df1.values.tolist())
                fin.append(df2)
                df1.drop(df.index, inplace=True)  # 清空df1
                df1 = pd.concat([df1, df])
            else:
                df1 = pd.concat([df1, df])

    print(labels)
    lengths = [len(i) for i in fin]
    print(lengths)
    pad_seq = pad_sequence(fin, batch_first=True)
    # print(pad_seq)
    print(pad_seq.shape)
    # f = open("test.txt", "w")
    # f.writelines(str(pad_seq))
    labelencoder = LabelEncoder()
    labels = labelencoder.fit_transform(labels)
    data = pad_seq.numpy()
    num_classses = max(labels) + 1
    data = {'X': data,
            'label': labels,
            'num_classes': num_classses,
            'lengths': lengths, }
    # 'num_words': len(my_vocab)}
    io.savemat('./data.mat', data)
    print('ok')


def plot_acc(train_acc):
    sns.set(style='darkgrid')
    plt.figure(figsize=(10, 7))
    x = list(range(len(train_acc)))
    plt.plot(x, train_acc, alpha=0.9, linewidth=2, label='train acc')
    plt.xlabel('Epoch')
    plt.ylabel('Acc')
    plt.legend(loc='best')
    plt.savefig('results/acc.png', dpi=400)


def plot_loss(train_loss):
    sns.set(style='darkgrid')
    plt.figure(figsize=(10, 7))
    x = list(range(len(train_loss)))
    plt.plot(x, train_loss, alpha=0.9, linewidth=2, label='train loss')
    plt.xlabel('Epoch')
    plt.ylabel('loss')
    plt.legend(loc='best')
    plt.savefig('results/loss.png', dpi=400)


class Data(Dataset):
    def __init__(self, mode='train'):
        data = io.loadmat('./data.mat')
        self.X = data['X']
        self.y = data['label']
        self.lengths = data['lengths']
        # self.num_words = data['num_words'].item()
        train_X, val_X, train_y, val_y, train_length, val_length = train_test_split(self.X, self.y.squeeze(),
                                                                                    self.lengths.squeeze(),
                                                                                    test_size=0.4, random_state=42)
        val_X, test_X, val_y, test_y, val_length, test_length = train_test_split(val_X, val_y, val_length,
                                                                                 test_size=0.5, random_state=42)
        test_X = self.X
        test_y = self.y.squeeze()
        test_length = self.lengths.squeeze()
        if mode == 'train':
            self.X = train_X
            self.y = train_y
            self.lengths = train_length
        elif mode == 'val':
            self.X = val_X
            self.y = val_y
            self.lengths = val_length
        elif mode == 'test':
            self.X = test_X
            self.y = test_y
            self.lengths = test_length

    def __getitem__(self, item):
        return self.X[item], self.y[item], self.lengths[item]

    def __len__(self):
        return self.X.shape[0]


class getDataLoader():
    def __init__(self, batch_size):
        train_data = Data('train')
        val_data = Data('val')
        test_data = Data('test')
        # print('test_data',test_data)
        self.traindl = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
        self.valdl = DataLoader(val_data, batch_size=batch_size, shuffle=True, num_workers=4)
        self.testdl = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=4)
        # self.num_words = train_data.num_words


class GRUWithAttention(nn.Module):
    def __init__(self, num_classes=2, input_size=14, hidden_dim=32, num_layers=2):
        super(GRUWithAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.attention = nn.Linear(hidden_dim * 2, 1)
        self.classification = nn.Sequential(
            nn.Linear(hidden_dim * 2, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, num_classes)
        )

    def forward(self, x, lengths):
        packed_input = pack_padded_sequence(x, lengths=lengths, batch_first=True, enforce_sorted=False)
        packed_output, hn = self.gru(packed_input)
        padded_output, _ = pad_packed_sequence(packed_output, batch_first=True)

        # Attention mechanism
        attn_weights = F.softmax(self.attention(padded_output), dim=1)
        attn_output = torch.sum(attn_weights * padded_output, dim=1)

        pred = self.classification(attn_output)
        return pred


# 定义训练过程
class Trainer():
    def __init__(self):
        safeCreateDir('results/')
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.save_model_address = '../USV_data/model/gru.pt'
        self.load_model_address = '../USV_data/model/gru.pt'
        self.save_excel = '../USV_data/input/usv_test.xlsx'
        self.save_csv_address = '../USV_data/output/output.csv'
        self._init_data()
        self._init_model()
        # self.dataset = "test"

    def _init_data(self):
        data = getDataLoader(batch_size=64)
        self.traindl = data.traindl
        self.valdl = data.valdl
        self.testdl = data.testdl
        # self.num_words = data.num_words

    def _init_model(self):
        self.net = GRUWithAttention(2).to(self.device)
        self.opt = Adam(self.net.parameters(), lr=1e-4, weight_decay=5e-4)
        self.cri = nn.CrossEntropyLoss()

    def save_model(self):
        # torch.save(self.net.state_dict(), '../USV_data/model/gru.pt')
        torch.save(self.net.state_dict(), self.save_model_address)


    def load_model(self):
        # self.net.load_state_dict(torch.load('../USV_data/model/gru.pt', map_location=torch.device('cpu')))  # 对选择用的模型
        self.net.load_state_dict(torch.load(self.load_model_address, map_location=torch.device('cpu')))  # 对选择用的模型

    def train(self, epochs):
        patten = 'Epoch: %d   [===========]  cost: %.2fs;  loss: %.4f;  train acc: %.4f;  val acc:%.4f;'
        train_accs = []
        c_loss = []
        for epoch in range(epochs):
            cur_preds = np.empty(0)
            cur_labels = np.empty(0)
            cur_loss = 0
            start = time.time()
            for batch, (inputs, targets, lengths) in enumerate(self.traindl):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                lengths = lengths.to('cpu')
                pred = self.net(inputs, lengths)
                loss = self.cri(pred, targets)
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                cur_preds = np.concatenate([cur_preds, pred.cpu().detach().numpy().argmax(axis=1)])
                cur_labels = np.concatenate([cur_labels, targets.cpu().numpy()])
                cur_loss += loss.item()
            acc, precision, f1, recall = metrics(cur_preds, cur_labels)
            val_acc, val_precision, val_f1, val_recall = self.val()
            train_accs.append(acc)
            c_loss.append(cur_loss)
            end = time.time()
            print(patten % (epoch, end - start, cur_loss, acc, val_acc))
            self.save_model()
            trainer.test()

        self.save_model()
        plot_acc(train_accs)
        plot_loss(c_loss)

    # @torch.no_grad()中的数据不需要计算梯度，也不会进行反向传播
    @torch.no_grad()
    def val(self):
        self.net.eval()
        cur_preds = np.empty(0)
        cur_labels = np.empty(0)
        for batch, (inputs, targets, lengths) in enumerate(self.valdl):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            lengths = lengths.to('cpu')
            pred = self.net(inputs, lengths)
            cur_preds = np.concatenate([cur_preds, pred.cpu().detach().numpy().argmax(axis=1)])
            cur_labels = np.concatenate([cur_labels, targets.cpu().numpy()])
        acc, precision, f1, recall = metrics(cur_preds, cur_labels)
        self.net.train()
        return acc, precision, f1, recall

    @torch.no_grad()
    def test(self):
        print("test ...")
        self.load_model()
        patten = 'test acc: %.4f   precision: %.4f   recall: %.4f    f1: %.4f    '
        self.net.eval()
        cur_preds = np.empty(0)
        cur_labels = np.empty(0)
        ns = np.empty(0)
        for batch, (inputs, targets, lengths) in enumerate(self.testdl):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            lengths = lengths.to('cpu')
            pred = self.net(inputs, lengths)
            n = inputs[:, 0, -1]
            n_cpu = n.cpu()
            ns = np.concatenate([ns, n_cpu])
            cur_preds = np.concatenate([cur_preds, pred.cpu().detach().numpy().argmax(axis=1)])

            cur_labels = np.concatenate([cur_labels, targets.cpu().numpy()])
        acc, precision, f1, recall = metrics(cur_preds, cur_labels)
        ns = [str(num).replace(' ', ',') for num in ns]
        cur_preds = [str(num).replace(' ', ',') for num in cur_preds]
        print('n')
        print(ns)
        print('预测值')
        print(cur_preds)
        df = pd.DataFrame({
            'USVs_per_segment': ns,
            'predicted_results': cur_preds})

        # df.to_csv('../USV_data/output/output.csv', index=False)
        df.to_csv(self.save_csv_address, index=False)


        df = pd.read_csv(self.save_csv_address)  # 替换为你的CSV文件名


        # 确保数据在正确的列中，假设第一列和第二列包含数值数据
        column1 = df.iloc[:, 0]  # 第一列
        column2 = df.iloc[:, 1]  # 第二列

        # 计算对应元素的乘积
        products = column1 * column2

        # 计算所有乘积的和
        sum_of_products = products.sum()
        print(sum_of_products)
        print("Proportion of mating USV")
        sum_n = column1.sum()
        percentages = (sum_of_products / sum_n) * 100
        print(percentages)
        # with open('more_output1.py', 'r', encoding='utf-8') as file:
        #     exec(file.read())
        # with open('more_output2.py', 'r', encoding='utf-8') as file:
        #     exec(file.read())
        # # with open('../USV_data/output/output_USV_Mating proportion.txt', 'w') as file:
        # #     file.write('Proportion of mating ultrasound:' + str(percentages) + '%')
        self.net.train()



if __name__ == "__main__":
    pre_read_excel_address = r'..\USV_data\input\usv_test.xlsx'
    getFormatData(pre_read_excel_address)  # 数据预处理：数据清洗和词向量读要预测数据
    trainer = Trainer()
    trainer.save_model_address = r'D:\Projects\all_model\video_model\USV_data\model\gru.pt'
    trainer.load_model_address = r'D:\Projects\all_model\video_model\USV_data\model\gru.pt'
    trainer.save_csv_address = r"D:\Projects\all_model\video_model\USV_data" + "\output.csv"

    # trainer.dataset = "test"
    # trainer.train(epochs=100)  # 数据训练

    trainer.test()  # 预测









































































































































































































































































































































































































































































































































































    