import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.autograd import Variable
from PIL import Image
import torch.nn.functional as F
from torch.nn import Parameter
import math

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

class ArcMarginProduct(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin
            cos(theta + m)
        """
    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cosine.size(), device='cuda')
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s
        # print(output)

        return output

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
    model.top_layer = nn.Linear(4096, 257)
    #model.top_layer = ArcMarginProduct(4096, 1222, s=30, m=0.5, easy_margin=False)

    #grayscale = nn.Conv2d(3, 1, kernel_size=1, stride=1, padding=0)
    #grayscale.weight.data.fill_(1.0 / 3.0)
    #grayscale.bias.data.zero_()
    #sobel_filter = nn.Conv2d(1, 2, kernel_size=3, stride=1, padding=1)
    #sobel_filter.weight.data[0,0].copy_(
    #    torch.FloatTensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    #    )
    #sobel_filter.weight.data[1,0].copy_(
    #    torch.FloatTensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    #    )
    #sobel_filter.bias.data.zero_()
    #model.sobel = nn.Sequential(grayscale, sobel_filter)
    #for p in model.sobel.parameters():
    #    p.requires_grad = False

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        if self.top_layer:
            x = self.top_layer(x)
        return x

    model.cuda()

    #  transform = transforms.Compose([
    #    transforms.Resize((224, 224), interpolation=Image.BICUBIC),
    #    transforms.ToTensor()
    #])
    #train_set = datasets.CIFAR100(root='../data', train=True, transform=transform, download=True)
    #test_set = datasets.CIFAR100(root='../data', train=False, transform=transform, download=True)
    #batch_size = 50
    #train_loader = torch.utils.data.DataLoader(
    #    train_set, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)
    #test_loader = torch.utils.data.DataLoader(
    #    test_set, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)

    traindir = ('/faces_83/train_images_s')
    testdir = ('/faces_83/test_images_s')
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

    epoch_num = 200 #200
    for epoch in range(epoch_num):
        print('epoch :', epoch)
        trainer.train_epoch()
        test(model, test_loader)
        # save running checkpoint
        #torch.save({'epoch': epoch + 1,
        #            'arch': 'vgg16',
        #            'state_dict': model.state_dict(),
        #            'optimizer' : optimizer.state_dict()},
        #            'checkpoint.pth.tar')
        torch.save(model.state_dict(), './checkpoint_s.tar')

        # save cluster assignments
        #cluster_log.log(deepcluster.images_lists)
