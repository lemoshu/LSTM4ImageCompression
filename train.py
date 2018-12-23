import time
import os
import argparse

import numpy as np

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as LS
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data as data
from torchvision import transforms

parser = argparse.ArgumentParser()
parser.add_argument(
    '--batch-size', '-N', type=int, default=32, help='batch size')
parser.add_argument(
    '--train', '-f', required=True, type=str, help='folder of training images')
parser.add_argument(
    '--max-epochs', '-e', type=int, default=200, help='max epochs')
parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
# parser.add_argument('--cuda', '-g', action='store_true', help='enables cuda')
parser.add_argument(
    '--iterations', type=int, default=16, help='unroll iterations')
parser.add_argument('--checkpoint', type=int, help='unroll iterations')
args = parser.parse_args()

## load 32x32 patches from images
import dataset

train_transform = transforms.Compose([
    transforms.RandomCrop((32, 32)),#把图片随机切分成32*32
    transforms.ToTensor(),
])

train_set = dataset.ImageFolder(root=args.train, transform=train_transform)

train_loader = data.DataLoader(
    dataset=train_set, batch_size=args.batch_size, shuffle=True, num_workers=1)

print('total images: {}; total batches: {}'.format(
    len(train_set), len(train_loader)))

## load networks on GPU
import network

encoder = network.EncoderCell().cuda()
binarizer = network.Binarizer().cuda()
decoder = network.DecoderCell().cuda()

solver = optim.Adam(
    [
        {
            'params': encoder.parameters()
        },
        {
            'params': binarizer.parameters()
        },
        {
            'params': decoder.parameters()
        },
    ],
    lr=args.lr)#定义了优化器是Adam，学习率是上面定义的0.005


def resume(epoch=None):
    if epoch is None:
        s = 'iter'
        epoch = 0
    else:
        s = 'epoch'

    encoder.load_state_dict(
        torch.load('checkpoint/encoder_{}_{:08d}.pth'.format(s, epoch)))
    binarizer.load_state_dict(
        torch.load('checkpoint/binarizer_{}_{:08d}.pth'.format(s, epoch)))
    decoder.load_state_dict(
        torch.load('checkpoint/decoder_{}_{:08d}.pth'.format(s, epoch)))


def save(index, epoch=True):
    if not os.path.exists('checkpoint'):
        os.mkdir('checkpoint')

    if epoch:
        s = 'epoch'
    else:
        s = 'iter'

    torch.save(encoder.state_dict(), 'checkpoint/encoder_{}_{:08d}.pth'.format(
        s, index))

    torch.save(binarizer.state_dict(),
               'checkpoint/binarizer_{}_{:08d}.pth'.format(s, index))

    torch.save(decoder.state_dict(), 'checkpoint/decoder_{}_{:08d}.pth'.format(
        s, index))


# resume()

scheduler = LS.MultiStepLR(solver, milestones=[3, 10, 20, 50, 100], gamma=0.5)

last_epoch = 0
if args.checkpoint:
    resume(args.checkpoint)
    last_epoch = args.checkpoint
    scheduler.last_epoch = last_epoch - 1

for epoch in range(last_epoch + 1, args.max_epochs + 1):

    scheduler.step()

    for batch, data in enumerate(train_loader):
        batch_t0 = time.time()

        ## init lstm state
        encoder_h_1 = (Variable(torch.zeros(data.size(0), 256, 8, 8).cuda()),
                       Variable(torch.zeros(data.size(0), 256, 8, 8).cuda()))
        encoder_h_2 = (Variable(torch.zeros(data.size(0), 512, 4, 4).cuda()),
                       Variable(torch.zeros(data.size(0), 512, 4, 4).cuda()))
        encoder_h_3 = (Variable(torch.zeros(data.size(0), 512, 2, 2).cuda()),
                       Variable(torch.zeros(data.size(0), 512, 2, 2).cuda()))

        decoder_h_1 = (Variable(torch.zeros(data.size(0), 512, 2, 2).cuda()),
                       Variable(torch.zeros(data.size(0), 512, 2, 2).cuda()))
        decoder_h_2 = (Variable(torch.zeros(data.size(0), 512, 4, 4).cuda()),
                       Variable(torch.zeros(data.size(0), 512, 4, 4).cuda()))
        decoder_h_3 = (Variable(torch.zeros(data.size(0), 256, 8, 8).cuda()),
                       Variable(torch.zeros(data.size(0), 256, 8, 8).cuda()))
        decoder_h_4 = (Variable(torch.zeros(data.size(0), 128, 16, 16).cuda()),
                       Variable(torch.zeros(data.size(0), 128, 16, 16).cuda()))

        patches = Variable(data.cuda())

        solver.zero_grad() # pytorch accumulates gradients, making them zero for each patch 对所有的参数的梯度缓冲区进行归零

        losses = []

        res = patches - 0.5

        bp_t0 = time.time()

        for _ in range(args.iterations):
            encoded, encoder_h_1, encoder_h_2, encoder_h_3 = encoder(
                res, encoder_h_1, encoder_h_2, encoder_h_3)

            codes = binarizer(encoded)

            output, decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4 = decoder(
                codes, decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4)

            res = res - output#caculate residual
            losses.append(res.abs().mean())#mean(abs(residual))

        bp_t1 = time.time()
 # forward + backward + optimize
        loss = sum(losses) / args.iterations #caculate the abs-error per iteration.(代价函数就是平均每次迭代的残差)
        loss.backward()#可以调用 .backward()来自动计算出所有的梯度，反向传播，我们调用loss.backward()时整个图通过代价来进行区分，图中所有的变量都会以.grad来累积梯度。
        solver.step()#Using Adam 

        batch_t1 = time.time()

        print(
            '[TRAIN] Epoch[{}]({}/{}); Loss: {:.6f}; Backpropagation: {:.4f} sec; Batch: {:.4f} sec'.
            format(epoch, batch + 1,
                   len(train_loader), loss.data[0], bp_t1 - bp_t0, batch_t1 -
                   batch_t0))
        print(('{:.4f} ' * args.iterations +
               '\n').format(* [l.data[0] for l in losses]))

        index = (epoch - 1) * len(train_loader) + batch

        ## save checkpoint every 500 training steps
        if index % 500 == 0:
            save(0, False)

    save(epoch)
