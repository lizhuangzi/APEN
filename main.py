#!/usr/bin/python3
import argparse
import itertools
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from PIL import Image
import torch
import os
import torch.optim as optim
from net import srnet,vgg,attNet,srcorrectNet
import shutil
from utils.MyImageFolder import process
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
parser.add_argument('--checkpoint', type=str, default='./model', help='save model root')
parser.add_argument('--name', type=str, default='vgg16caltech256', help='train name')
parser.add_argument('--n_epochs', type=int, default=60, help='number of epochs of training')
parser.add_argument('--batchSize', type=int, default=16, help='size of the batches')
parser.add_argument('--dataroot', type=str, default='/lxbdata/caltech256R', help='root directory of the dataset')
parser.add_argument('--clr', type=float, default=0.001, help='initial classfi learning rate')
parser.add_argument('--slr', type=float, default=0.0001, help='initial sr learning rate')
parser.add_argument('--alr', type=float, default=0.0001, help='initial sr learning rate')
parser.add_argument('--weight-decay', '--wd', default=10 ** -5, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--size', type=int, default=256, help='size of the data crop (squared assumed)')
parser.add_argument('--num_classes', type=int, default=257, help='size of the num_classes')
parser.add_argument('--cuda', action='store_true', default='0', help='use GPU computation')
opt = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.cuda)
print(opt)
writer = SummaryWriter('runs/'+opt.name)

def AdjustLearningRate(optimizer, epoch, learning_rate):
    lr = learning_rate * (0.1 ** (epoch // 20))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer
def AdjustLearningRate1(optimizer, epoch, learning_rate):
    lr = learning_rate * (0.1 ** (epoch // 25))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

###### Definition of variables ######
# Networks
srmodel=srnet(16,4)
classifmodel=vgg(opt.num_classes)
attmodel=attNet()
srcorrectmodel=srcorrectNet()


if opt.cuda:
    srmodel.cuda()
    classifmodel.cuda()
    attmodel.cuda()
    srcorrectmodel.cuda()


# Lossess
criterion_G = torch.nn.MSELoss()
criterion_class = torch.nn.CrossEntropyLoss()
adversarial_loss = torch.nn.BCELoss()


state_dict = torch.load('model/train_caltech_vgg16_224lbn_best.pth', map_location=lambda storage, loc: storage)
classifmodel.load_state_dict(state_dict)



# Optimizers & LR schedulers
classoptimizer = optim.SGD(
        classifmodel.parameters(),
        lr=opt.clr,
        weight_decay=opt.weight_decay)
sroptimizer = optim.Adam(srmodel.parameters(), lr=opt.slr, betas=(0.9, 0.999), eps=1e-8,
                           weight_decay=opt.weight_decay)
attoptimizer = optim.Adam(attmodel.parameters(), lr=opt.alr, betas=(0.9, 0.999), eps=1e-8,
                           weight_decay=opt.weight_decay)
srcorroptimizer = optim.Adam(srcorrectmodel.parameters(), lr=opt.alr, betas=(0.9, 0.999), eps=1e-8,
                           weight_decay=opt.weight_decay)




### data processing
transform_train1 = transforms.Compose(
        [
             transforms.Scale(256),
             transforms.RandomCrop(224),
             transforms.RandomHorizontalFlip(),
        ])
transform_train2 = transforms.Compose(
        [
             transforms.ToTensor(),
             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
transform_train3 = transforms.Compose(
        [
            transforms.Scale(56),
        ])
transform_test1 = transforms.Compose(
        [
             transforms.Scale(256),
             transforms.CenterCrop(224),
        ])
transform_test2 = transforms.Compose(
        [
             transforms.ToTensor(),
             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
transform_test3 = transforms.Compose(
        [
             transforms.Scale(56),
        ])

train_path=os.path.join(opt.dataroot,'train')
test_path=os.path.join(opt.dataroot,'test')
trainset = process(train_path,transform=transform_train1,transform1=transform_train2,transform2=transform_train3)
train_loader = DataLoader(trainset, batch_size=opt.batchSize, shuffle=True)

testset = process(test_path,transform=transform_test1,transform1=transform_test2,transform2=transform_test3)
test_loader = DataLoader(testset, batch_size=opt.batchSize, shuffle=False)
best_acc=0


def test(epoch):
    srmodel.eval()
    classifmodel.eval()
    attmodel.eval()
    srcorrectmodel.eval()
    test_loss = 0
    top1 = 0
    top5 = 0
    top1sr = 0
    top5sr = 0
    for iter, (imhr, imlr, targets) in enumerate(test_loader):
        if opt.cuda:
            imhr, imlr, targets = imhr.cuda(), imlr.cuda(), targets.cuda()
        imhr, imlr, targets = Variable(imhr),Variable(imlr), Variable(targets)
        imsr,fsr=srmodel(imlr)
        att=attmodel(fsr)
        imsr=imsr*att
        srfeature = srcorrectmodel(imsr)
        outputshr, _, _, _ = classifmodel(x=imhr,y=0)
        outputssr, _, _, _ = classifmodel(x=imsr,y=srfeature)
        loss = criterion_class(outputshr, targets)
        test_loss += loss.item()
        prec1, prec5 = accuracy(outputshr.data, targets.data, topk=(1, 5))
        top1 += prec1
        top5 += prec5

        prec1sr, prec5sr = accuracy(outputssr.data, targets.data, topk=(1, 5))
        top1sr += prec1sr
        top5sr += prec5sr



        print('[epcho:%d][%d/%d]|Loss: %.3f '
              % (epoch, iter, len(test_loader),test_loss / (iter + 1)))
        print ('Top1: %.3f%%|Top5: %.3f%%' % (100 * top1 / (iter+1), 100 * top5 / (iter+1)))
        print ('Top1: %.3f%%|Top5: %.3f%%' % (100 * top1sr / (iter+1), 100 * top5sr / (iter+1)))

    acc = 100 * top1sr / len(test_loader)
    writer.add_scalars('data/testacc', {'HRacc': top1 / len(test_loader),
                                    'SRacc': top1sr / len(test_loader)}, epoch)
    writer.add_scalars('data/testTopacc', {'Top1': top1/ len(test_loader),
                                       'Top5': top5/ len(test_loader),
                                       'Top1sr': top1sr/ len(test_loader),
                                       'Top5sr': top5sr/ len(test_loader)}, epoch)
    return acc

def save_checkpoint(state, checkpoint,is_best,name=''):
    filepath = os.path.join(checkpoint, 'last_'+name+'.pth.tar')
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'best_'+name+'.pth.tar'))


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(1.0 / batch_size))
    return res
###### Training ######
for epoch in range(opt.epoch, opt.n_epochs):
    mse_loss=0.0
    labelmse_loss=0.0
    Gen_loss=0.0
    dis_loss=0.0
    cif_loss=0.0
    classification_loss=0.0
    attloss=0.0
    gloss=0
    srcloss=0
    total=0
    correct=0
    correctsr=0
    top1 = 0
    top5 = 0
    top1sr = 0
    top5sr = 0


    for iter, traindata in enumerate(train_loader):
        imhr, imlr, train_label= traindata
        if opt.cuda:
            imhr, imlr, train_label = imhr.cuda(), imlr.cuda(), train_label.cuda(async=True)
        else:
            imhr, imlr, train_label = Variable(imhr), Variable(imlr), Variable(train_label)
        ###### training srmodel######
        sroptimizer.zero_grad()
        imsr, fsr = srmodel(imlr)
        srloss = criterion_G(imsr, imhr)
        # srloss = 0
        mse_loss+=srloss

        _, c1sr, c2sr, c3sr= classifmodel(x=imsr,y=0)
        _, c1hr, c2hr, c3hr= classifmodel(x=imhr.detach(),y=0)
        classloss1 = criterion_G(c1sr, c1hr)
        classloss2 = criterion_G(c2sr, c2hr)
        classloss3 = criterion_G(c3sr, c3hr)
        classloss=classloss1+classloss2+classloss3
        labelmse_loss+=classloss



        Gloss=0.01*srloss+classloss

        Gen_loss+=Gloss

        Gloss.backward()
        sroptimizer.step()

        ###### train attention ######
        attoptimizer.zero_grad()
        att=attmodel(fsr.detach())
        cinput=imsr.detach()*att
        srl, _, _, _ = classifmodel(x=cinput,y=0)
        att_loss = criterion_class(srl, train_label)
        attloss+=att_loss
        att_loss.backward()
        attoptimizer.step()



        ###### train classification ######
        classoptimizer.zero_grad()
        clhr, _, _, _= classifmodel(x=imhr,y=0)

        class_loss = criterion_class(clhr, train_label)
        cif_loss+=class_loss

        Closs=class_loss
        classification_loss+=Closs



        Closs.backward()
        classoptimizer.step()


        ###### train srcorrectmodel ######
        srcorroptimizer.zero_grad()
        srfeature=srcorrectmodel(cinput.detach())
        src, _, _, _=classifmodel(x=cinput.detach(),y=srfeature)
        srclass_loss = criterion_class(src, train_label)
        srcloss+=srclass_loss
        srclass_loss.backward()
        srcorroptimizer.step()

        # measure accuracy and record loss
        prec1, prec5 = accuracy(clhr.data, train_label.data, topk=(1, 5))
        top1 += prec1
        top5 += prec5

        prec1sr, prec5sr = accuracy(src.data, train_label.data, topk=(1, 5))
        top1sr += prec1sr
        top5sr += prec5sr



        lr=0
        lrs=0
        for param_group in classoptimizer.param_groups:
            lr=param_group['lr']
        for param_group in srcorroptimizer.param_groups:
            lrs=param_group['lr']
        print('[epcho:%d][%d/%d][LR:%.5f|LRS:%.5f] MSEloss:%.3f|LabelMSEloss:%.3f||attloss:%.3f||Closs:%.3f||srcloss:%.3f'%(epoch,iter+1,len(train_loader),lr,lrs,mse_loss/(iter+1),labelmse_loss/(iter+1),attloss/(iter+1),cif_loss/(iter+1),srcloss/(iter+1)))
        print('[Totle loss] SRmodelloss:%.3f|attmodelloss:%.3f|classmodelloss:%.3f|srcmodelloss:%.3f'%(Gen_loss/(iter+1),attloss/(iter+1),classification_loss/(iter+1),srcloss/(iter+1)))
        print ('Top1: %.3f%%|Top5: %.3f%%'%(100*top1/(iter+1),100*top5/(iter+1)))
        print ('Top1: %.3f%%|Top5: %.3f%%' % (100*top1sr/(iter+1),100*top5sr/(iter+1)))


    writer.add_histogram('zz/x', Gen_loss / len(train_loader), epoch)
    writer.add_histogram('zz/y',classification_loss / len(train_loader), epoch)
    writer.add_scalars('data/loss', {'mesloss': mse_loss/len(train_loader),
                                       'lmseloss': labelmse_loss/len(train_loader),
                                       'attloss': attloss / len(train_loader),
                                       'cifloss': cif_loss / len(train_loader)}, epoch)
    writer.add_scalar('data/Gloss', Gen_loss / len(train_loader), epoch)
    writer.add_scalar('data/Closs', classification_loss / len(train_loader), epoch)
    writer.add_scalars('data/acc', {'HRacc': top1/len(train_loader),
                                             'SRacc': top1sr/len(train_loader)}, epoch)
    writer.add_scalars('data/Topacc', {'Top1': top1/len(train_loader),
                                             'Top5': top5/len(train_loader),
                                             'Top1sr': top1sr/len(train_loader),
                                             'Top5sr':top5sr/len(train_loader)}, epoch)
    writer.add_text('zz/text', 'zz: this is epoch ' + str(epoch), epoch)

    acc=test(epoch)
    is_best = acc > best_acc
    best_acc = max(acc, best_acc)
    # Update learning rates
    sroptimizer = AdjustLearningRate1(sroptimizer, epoch, opt.slr)
    attoptimizer = AdjustLearningRate1(attoptimizer, epoch, opt.alr)
    classoptimizer = AdjustLearningRate(classoptimizer, epoch, opt.clr)
    srcorroptimizer = AdjustLearningRate1(srcorroptimizer, epoch, opt.alr)
    save_checkpoint(srmodel.state_dict(), opt.checkpoint, is_best, name=opt.name+'sr')
    save_checkpoint(attmodel.state_dict(), opt.checkpoint, is_best, name=opt.name + 'att')
    save_checkpoint(classifmodel.state_dict(), opt.checkpoint, is_best, name=opt.name+'cl')
    save_checkpoint(srcorrectmodel.state_dict(), opt.checkpoint, is_best, name=opt.name + 'srcl')


writer.close()



###################################
