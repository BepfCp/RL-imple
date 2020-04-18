import torch
import torchvision
import torch.nn as nn
import torch.optim as opt
import torch.nn.functional as F
import torchvision.transforms as tfms

# 1. prepare data
tfm = tfms.Compose(  # https://pytorch.org/tutorials/beginner/data_loading_tutorial.html#compose-transforms
    [
        tfms.ToTensor(),
        # https://pytorch.org/docs/stable/torchvision/transforms.html#torchvision.transforms.Normalize
        tfms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
)
def data_set(is_train):
    return torchvision.datasets.CIFAR10(root='./data',
                                         train=is_train,
                                         download=False,
                                         transform=tfm)  # torchvision datasets are PILImage images of range [0, 1]

def data_loader(data_set, is_shuffle):
    # https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
    return torch.utils.data.DataLoader(data_set,
                                       batch_size = 4,
                                       shuffle = is_shuffle,
                                       num_workers = 0)  # carefully select num_workers!

train_set = data_set(True)  # https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
train_loader = data_loader(train_set, True)
test_set = data_set(False)
test_loader = data_loader(test_set, False)
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 2. define a neural network
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        # The input of a Pytorch Neural Network is of type
        # [BATCH_SIZE] * [CHANNEL_NUMBER] * [HEIGHT] * [WIDTH]
        # So resize the tensor before the first fully connected layer
        # See: https://discuss.pytorch.org/t/input-size-of-fc-layer-in-tutorial/14644
        x = x.view(-1, 16*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
net = Net()

# 3. define a loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = opt.SGD(net.parameters(), lr=0.001, momentum=0.9)

# 4. train the network
def training(epochs=2, num_print=1000):
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # net.to(device)
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):  # https://www.geeksforgeeks.org/enumerate-in-python/
            inputs, labels = data # inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()  # zero the parameter gradients
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.item()
            if i%num_print == num_print-1:
                print('[%d %5d] loss: %.3f' % (epoch+1, i+1, running_loss/num_print))
                running_loss = 0.0
    print("Finished Training!")

# training()

# 5. save the model
PATH = './cifar_net.pth'
# torch.save(net.state_dict(), PATH)

# 6. reload model
net = Net()
net.load_state_dict(torch.load(PATH))

# 7. get accuracy
correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)  # https://blog.csdn.net/Z_lbj/article/details/79766690
        total += labels.size(0)
        correct += (predicted==labels).sum().item()

print("Accuracy on the %d test images: %d %%"%(total, 100*correct/total))

# 8. get accuracy of every class
class_correct = [0 for i in range(10)]
class_total = [0 for i in range(10)]
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted==labels).squeeze()  # https://deeplizard.com/learn/video/fCVuiW9AFzY
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1
for i in range(10):
    print("Accuracy of %s: %d %%" % (classes[i], 100*class_correct[i]/class_total[i]))

if __name__ == '__main__':
    pass