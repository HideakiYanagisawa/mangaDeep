import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.autograd import Variable
from PIL import Image

class Trainer():
    def __init__(self, model, optimizer, train_loader):
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader

    def train_epoch(self):
        self.model.train()
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = Variable(data.cuda()), Variable(target.cuda())
            self.optimizer.zero_grad()
            output = self.model(data)
            criterion = nn.CrossEntropyLoss().cuda()
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            print('[{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))

def test(model, test_loader):
    model.eval()
    correct = 0
    for batch_idx, (data, target) in enumerate(test_loader):
        data, target = Variable(data.cuda(), requires_grad=False), Variable(target.cuda())
        output = model(data)
        pred = output.data.max(dim=1)[1]
        correct += pred.eq(target.data).cpu().sum()

    print('Accuracy: %d %%' % (
        100 * correct / len(test_loader.dataset)))


if __name__ == '__main__':

    model = models.vgg16(pretrained=True)
    for p in model.features[:15].parameters():
        p.requires_grad = False
    model.classifier = nn.Sequential(
        nn.Linear(512 * 7 * 7, 4096),
        nn.ReLU(True),
        nn.Dropout(0.5),
        nn.Linear(4096, 4096),
        nn.ReLU(True),
    )
    model.top_layer = nn.Linear(4096, 1222)
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        if self.top_layer:
            x = self.top_layer(x)
        return x

    model.cuda()

    traindir = ('/faces_83/train_images')
    testdir = ('/faces_83/test_images')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=64, shuffle=(
            train_sampler is None),
        num_workers=0, pin_memory=True, sampler=train_sampler)

    test_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(testdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=64, shuffle=False,
        num_workers=0, pin_memory=True)

    optimizer = optim.SGD(model.classifier.parameters(), lr=0.001, momentum=0.5)
    trainer = Trainer(model, optimizer, train_loader)

    epoch_num = 3 #200
    for epoch in range(epoch_num):
        print('epoch :', epoch)
        trainer.train_epoch()
        test(model, test_loader)
        torch.save(model.state_dict(), './checkpoint_test.tar')
