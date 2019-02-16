import torch
from data import *
from model import *
import random
import time
import math

n_hidden = 128
n_epochs = 10000
print_every = 500
learning_rate = 0.001 # If you set this too high, it might explode. If too low, it might not learn

def categoryFromOutput(output):
    top_n, top_i = output.data.topk(1) # Tensor out of Variable with .data
    category_i = top_i[0][0]
    return all_categories[category_i], category_i

def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]

def randomTrainingPair():
    category = randomChoice(all_categories)
    line = randomChoice(category_lines[category])
    category_tensor = Variable(torch.LongTensor([all_categories.index(category)]))
    line_tensor = Variable(lineToTensor(line))
    return category, line, category_tensor, line_tensor

rnn = RNN(n_letters, n_hidden, n_categories)
optimizer = torch.optim.SGD(rnn.parameters(), lr=learning_rate)
criterion = nn.NLLLoss()

useCuda = True

if useCuda:
    print('GPU available')
    print(torch.cuda.get_device_name(0))
    rnn = rnn.cuda()

def train(category_tensor, line_tensor):
    hidden = rnn.initHidden()
    optimizer.zero_grad()
    if useCuda:
        category_tensor = Variable(category_tensor).cuda()
        line_tensor = Variable(line_tensor).cuda()
        hidden = Variable(hidden).cuda()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    loss = criterion(output, category_tensor)
    loss.backward()

    optimizer.step()

    # print(loss.data.item())

    return output, loss.data.item()

# Keep track of losses for plotting
current_loss = 0
all_losses = []

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

start = time.time()
correct_count = 0

for epoch in range(1, n_epochs + 1):
    category, line, category_tensor, line_tensor = randomTrainingPair()
    output, loss = train(category_tensor, line_tensor)
    current_loss += loss
    guess, guess_i = categoryFromOutput(output)
    if guess == category:
        correct_count = correct_count + 1
    
    # Print epoch number, loss, name and guess
    if epoch % print_every == 0:
        print('%d %d%% (%s) %.4f / accuracy: %.4f' % (epoch, epoch / n_epochs * 100, timeSince(start), current_loss/print_every, correct_count/print_every))
        all_losses.append(current_loss / print_every)
        current_loss = 0
        correct_count = 0

torch.save(rnn, 'rnn-classification.pt')

