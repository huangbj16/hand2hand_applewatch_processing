import torch
from torch import nn
from data import *
from sklearn.model_selection import train_test_split, ShuffleSplit
import time
import matplotlib.pyplot as plt

print(n_categories)
print('categories: ', all_categories)

#const number
n_seqlen = 50
n_inputsize = 18
n_hidden = 128
n_epochs = 2000
learning_rate = 0.1 # If you set this too high, it might explode. If too low, it might not learn

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.lstm = nn.LSTM(n_inputsize, n_hidden, batch_first = True)
        self.fc = nn.Linear(n_hidden, n_categories)
        self.logsoftmax = nn.LogSoftmax()

    def forward(self, x):
        # print(x.shape)
        x = torch.from_numpy(x.astype('float32'))
        x, (hn, cn) = self.lstm(x)
        x = x[:, -1, :]
        x = x.view(-1, 1, n_hidden)
        # print(x.shape)
        # 取最后一个数据
        x = self.fc(x)
        x = x.view(-1, n_categories)
        # print(x.shape)
        x = self.logsoftmax(x)
        # print(x.shape)
        return x # 11 dims

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
seed = int(time.time()*10000000) % 19980608
X_train, X_test, y_train, y_test = train_test_split(feature_set, flag_set, test_size=0.2, random_state=seed)
print('split result shape: ', X_train.shape, X_test.shape, y_train.shape, y_test.shape)

valid_result = []
test_result = []

for epoch in range(n_epochs):
    #train
    optimizer.zero_grad()
    # 每次随机1/10的train样本作为一个epoch
    seed = int(time.time()*10000000) % 19980608
    epoch_X_train, epoch_X_test, epoch_y_train, epoch_y_test = train_test_split(X_train, y_train, test_size=0.9, random_state=seed)
    # 训练
    epoch_X_train = epoch_X_train.reshape(-1, 50, 18)

    output = net(epoch_X_train)

    epoch_y_train = torch.LongTensor(epoch_y_train)
    loss = criterion(output, epoch_y_train)
    # print(loss)
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:

        #验证
        X_valid = X_train.reshape(-1, 50, 18)
        output = net(X_valid)
        count = 0
        for i in range(y_train.shape[0]):
            if y_train[i] == np.argmax(output.data.numpy()[i]):
                count = count + 1
        print(count/y_train.shape[0])
        valid_result.append(count/y_train.shape[0])
        #测试
        X_test = X_test.reshape(-1, 50, 18)
        output = net(X_test)
        count = 0
        for i in range(y_test.shape[0]):
            if y_test[i] == np.argmax(output.data.numpy()[i]):
                count = count + 1
        print(count/y_test.shape[0])
        test_result.append(count/y_test.shape[0])

        if epoch % 100 == 0:
            print(epoch)
        #     plt.plot(valid_result)
        #     plt.plot(test_result)
        #     plt.show()

plt.plot(valid_result)
plt.plot(test_result)
plt.show()

torch.save(net, 'lstm-classification.pt')