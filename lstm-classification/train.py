import torch
from torch import nn
from data import *
from sklearn.model_selection import train_test_split, ShuffleSplit
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import time
import matplotlib.pyplot as plt

print(n_categories)
print('categories: ', all_categories)
 
#const number
n_seqlen = 50
n_inputsize = 20
n_hidden = 128
n_epochs = 5000
learning_rate = 0.01 # If you set this too high, it might explode. If too low, it might not learn
n_freq_in = 200
n_freq_out = 128

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.lstm = nn.LSTM(n_inputsize, n_hidden, batch_first = True).cuda()
        self.fc = nn.Linear(n_hidden, n_categories).cuda()
        self.logsoftmax = nn.LogSoftmax().cuda()
        # self.fc1 = nn.Linear(52, 20).cuda()
        # self.relu = nn.ReLU().cuda()

    def forward(self, x):
        # y = x[:, 0:52]
        # x = x[:, 52:52+1000].reshape(-1, 50 ,20)
    
        x = x.reshape(-1, 50, 20)
        x = torch.from_numpy(x.astype('float32')).cuda()
        x, (hn, cn) = self.lstm(x)
        x = torch.max(x, dim = 1)[0]
        
        # y = torch.from_numpy(y.astype('float32')).cuda()
        # y = self.fc1(y)
        # y = self.relu(y)

        # x = torch.cat([x, y], 1)
        x = self.fc(x)

        x = x.view(-1, n_categories)
        x = self.logsoftmax(x)
        return x

start = 0
test_segment = [0]

for suffix_index in range(len(suffixes)):
    start = start + np.sum([category_lines[suffix_index*n_categories+i].shape[0] for i in range(n_categories)])
    test_segment.append(start)

print(test_segment)

tot_res_valid = []
tot_res_test = []

for suffix_index in range(len(suffixes)):
    print(suffixes[suffix_index])

    feature_train_set = np.concatenate((feature_set[: test_segment[suffix_index]], feature_set[test_segment[suffix_index+1]: ]))
    flag_train_set = np.concatenate((flag_set[: test_segment[suffix_index]], flag_set[test_segment[suffix_index+1]: ]))
    feature_test_set = feature_set[test_segment[suffix_index]: test_segment[suffix_index+1]] 
    flag_test_set = flag_set[test_segment[suffix_index]: test_segment[suffix_index+1]]
    print(feature_train_set.shape, flag_train_set.shape, feature_test_set.shape, flag_test_set.shape)

    net = Net()
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()
    # input = torch.randn(3, n_seqlen, n_inputsize)
    # output, (hn, cn) = rnn(input)
    # print(output.size())
    # input的维度：batch，数据的个数，单个数据的长度；
    # output的维度：batch，数据的个数，隐藏层的维度；（在此任务中，应该只取最后一个隐藏层）
    # 还需要一层全相连

    # 切割数据集
    # seed = int(time.time()*10000000) % 19980608
    # X_train, X_test, y_train, y_test = train_test_split(feature_set, flag_set, test_size=0.2, random_state=seed)
    # print('split result shape: ', X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    X_train, X_test, y_train, y_test = feature_train_set, feature_test_set, flag_train_set, flag_test_set

    valid_result = []
    test_result = []

    for epoch in range(n_epochs):
        #train
        optimizer.zero_grad()
        # 每次随机1/10的train样本作为一个epoch
        seed = int(time.time()*10000000) % 19980608
        epoch_X_train, epoch_X_test, epoch_y_train, epoch_y_test = train_test_split(X_train, y_train, test_size=0.9, random_state=seed)
        # epoch_X_train = X_train
        # epoch_y_train = y_train
        # 训练
        output = net(epoch_X_train)
        
        epoch_y_train = (torch.LongTensor(epoch_y_train)).cuda()
        loss = criterion(output, epoch_y_train)
        # print(loss)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            predict_temp = []
            label_temp = []
            #验证
            output = net(X_train).cpu()
            for i in range(y_train.shape[0]):
                if y_train[i] != 8:
                    predict_temp.append(np.argmax(output.data.numpy()[i]))
                    label_temp.append(y_train[i])
            valid_result.append([accuracy_score(label_temp, predict_temp), precision_score(label_temp, predict_temp, average='weighted'), recall_score(label_temp, predict_temp, average='weighted'), f1_score(label_temp, predict_temp, average='weighted')])
            #测试
            predict_temp = []
            label_temp = []
            output = net(X_test).cpu()
            for i in range(y_test.shape[0]):
                if y_test[i] != 8:
                    predict_temp.append(np.argmax(output.data.numpy()[i]))
                    label_temp.append(y_test[i])
            test_result.append([accuracy_score(label_temp, predict_temp), precision_score(label_temp, predict_temp, average='weighted'), recall_score(label_temp, predict_temp, average='weighted'), f1_score(label_temp, predict_temp, average='weighted')])

            if epoch % 100 == 0:
                print(epoch, valid_result[-1], test_result[-1])
                # plt.plot(valid_result)
                # plt.plot(test_result)
                # plt.show()

    # plt.plot(valid_result)
    # plt.plot(test_result)
    # plt.show()
    valid_result = np.array(valid_result)
    test_result = np.array(test_result)
    print(np.max(valid_result, axis=0))
    print(np.max(test_result, axis=0))
    tot_res_valid.append(np.max(valid_result, axis=0))
    tot_res_test.append(np.max(test_result, axis=0))
    
    torch.save(net, 'lstm-classification.pt')

print(tot_res_valid)
print(tot_res_test)