from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import torch.backends.cudnn as cudnn
from data import data_dataset# , data_noise_dataset, distilled_dataset
#from models import resnet_transition
from models import resnet, wideresnet, vggnet
import numpy as np
from torch.autograd import Variable
import torch.optim as optim
import logging
import random
import copy
import math
import itertools
import warnings
import info_nce
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from tsne_torch import tsne
from timeit import default_timer as timer
from autoattack import AutoAttack

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='PyTorch CIFAR TRADES Adversarial Training')
parser.add_argument('--epsilon', type=float, default=8.0/255.0, help='perturb scale')
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')
parser.add_argument('--batch-size', type=int, default=500, metavar='N',
                    help='input batch size for training (default: 500)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--dataset', type=str,
                    help='fmnist,cifar10,svhn', default='cifar10')
parser.add_argument('--model-dir', default='./checkpoint/encoder_C/TRADES_train',#CE_INY_0.33_0.2_scn9_sn2
                    help='directory of model for saving checkpoint')
parser.add_argument('--nat-img-train', type=str,
                    help='natural training data', default='./data/train_images.npy')
parser.add_argument('--nat-label-train', type=str,
                    help='natural training label', default='./data/train_labels.npy')
parser.add_argument('--nat-img-test', type=str,
                    help='natural test data', default='./data/test_images.npy')
parser.add_argument('--nat-label-test', type=str,
                    help='natural test label', default='./data/test_labels.npy')
parser.add_argument('--num-steps', default=10,
                    help='perturb number of steps')
parser.add_argument('--step-size', default=0.007,
                    help='perturb step size')
parser.add_argument('--p', type=float, default=0.05)
parser.add_argument('--sample_class_number', type=int, default=1, metavar='N',
                    help='how many neg classes are there when sampling for one anchor sample')
parser.add_argument('--sample_number', type=int, default=1,
                    help='how many neg samples are there when sampling in one neg class')
parser.add_argument('--alpha', type=float, default=0, help='weight for adv loss')
parser.add_argument('--beta', type=float, default=6.0, help='weight for adv loss')
parser.add_argument('--gamma', type=float, default=0, help='weight for neg adv loss')
parser.add_argument('--delta', type=float, default=0, help='weight for infoNCE loss')
parser.add_argument('--zeta', type=float, default=13.0, help='weight for neg avg adv loss')
parser.add_argument('--temp', type=float, default=1.0, help='tempurature for loss')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
use_cuda = not args.no_cuda and torch.cuda.is_available()
sn = args.sample_number
scn = args.sample_class_number
alpha = args.alpha
beta = args.beta
gamma = args.gamma
delta = args.delta
zeta = args.zeta
temp = args.temp

model_dir = './checkpoint/encoder_C/Trades_cos_detach_nat_train-adv_loss-6.0-neg_avg_adv_loss-detach-nat-0.015-scn-1-sn-1'
#Trades_mse_detach_nat_train-adv_loss-6.0-neg_avg_adv_loss-detach-nat-0.1-scn-1-sn-1
#Trades_cos_train-adv_loss-6.0-neg_avg_adv_loss-detach-nat-1.0-scn-1-sn-1
#Trades_cos_train-adv_loss-6.0-neg_avg_adv_loss-0.25-scn-1-sn-1
#Trades_mse__train-adv_loss-6.0-neg_avg_adv_loss-detach-nat-0.001-scn-1-sn-1
#Trades_pgd+kl_train-adv_loss-6.0-neg_avg_adv_loss-detach-nat-13.0-scn-1-sn-1
#Trades_pgd+kl+kl_pgd_train-adv_loss-6.0-neg_avg_adv_loss-detach-nat-13.0-scn-1-sn-1
#'Trades_train-adv_loss-6.0-neg_avg_adv_loss-detach-nat-13.0-scn-1-sn-1'
#Trades_trainuniform_alpha0.16666666666666666-adv_loss-6.0-neg_avg_adv_loss-13.0-scn-1-sn-1
#'./checkpoint/encoder/MART_train-adv_loss-5.0-neg_avg_adv_loss-15.0-scn-1-sn-1'
#encoder_MART/MART_resnet_18/
#encoder_C/TRADES_train-adv_loss_-6.0-neg_avg_adv_loss-13.0-scn-1-sn-1
#neg_train-neg_ce_loss-0.0-adv_loss-6.0-neg_adv_loss-0.0-neg_avg_adv_loss-0.0-infonce_loss-0.0-scn-1-sn-1-delay_epoch-0



logger = logging.getLogger(__name__)
logfile = os.path.join(model_dir, 'test.log')
logging.basicConfig(
    format='[%(asctime)s] - %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S',
    level=logging.INFO,
    filename=os.path.join(model_dir, 'test.log'))
logger.info(args)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)






def lode_feature_align(loader, model):
    for iter, (data, label) in enumerate(loader):
        data1 = data.cuda()
        data2 = u_pgd(model=model, x_natural=data.cuda(), y=label.cuda(), step_size=args.step_size,
                      epsilon=args.epsilon, perturb_steps=args.num_steps)
        data3 = torch.Tensor(np.random.permutation(data1.cpu().numpy())).cuda()
        _, feature1 = model(data1, prejection=True)
        _, feature2 = model(data2, prejection=True)
        _, feature3 = model(data3, prejection=True)
        if iter == 0:
            dis_adv = lalign(feature1.cpu(), feature2.cpu())
            dis_nat = lalign(feature1.cpu(), feature3.cpu())
            al_adv = dis_adv.sum() / data.shape[0]
            al_nat = dis_nat.sum() / data.shape[0]
        else:
            dis_adv = torch.cat((dis_adv, lalign(feature1.cpu(), feature2.cpu())), dim=0)
            al_adv += dis_adv.sum() / data.shape[0]
            dis_nat = torch.cat((dis_nat, lalign(feature1.cpu(), feature3.cpu())), dim=0)
            al_nat += dis_nat.sum() / data.shape[0]
        #if iter >= 0.5 * len(loader):
        #    break
    return al_adv / len(loader), dis_adv, al_nat / len(loader), dis_nat




def lode_feature_uniform(loader, model):
    un1, un2, un3 = 0., 0., 0.
    for iter, (data, label) in enumerate(loader):
        data1 = data.cuda()
        data2 = u_pgd(model=model, x_natural=data.cuda(), y=label.cuda(), step_size=args.step_size,
                      epsilon=args.epsilon, perturb_steps=args.num_steps)

        _, feature1 = model(data1, prejection=True)
        _, feature2 = model(data2, prejection=True)

        feature = torch.cat((feature1.cpu(), feature2.cpu()), dim=0)
        un1 += lunif(feature.cpu()) / len(loader)
        un2 += lunif(feature1.cpu()) / len(loader)
        un3 += lunif(feature2.cpu()) / len(loader)
        #if iter >= 0.5 * len(loader):
        #    break
    return un1.log(), un2.log(), un3.log()







def u_pgd(model, x_natural, y, step_size=0.003,
                epsilon=8.0/255.0, perturb_steps=10, distance='l_inf'):
    #imsave(x_natural[0], "x_out/%d" % i, "clean")
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()

    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                logits = model(x_adv)
                loss_kl = F.cross_entropy(logits, y)

            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    elif distance == 'l_2':
        batch_size = len(x_natural)
        delta = 0.001 * torch.randn(x_natural.shape).cuda().detach()
        delta = Variable(delta.data, requires_grad=True)

        # Setup optimizers
        optimizer_delta = optim.SGD([delta], lr=epsilon / perturb_steps * 2)

        for _ in range(perturb_steps):
            x_adv = x_natural + delta

            # optimize
            optimizer_delta.zero_grad()
            with torch.enable_grad():
                logits = model(x_adv)
                # logits = F.softmax(logits, dim=1)
                loss = F.cross_entropy(logits, y)
            loss.backward()
            # renorming gradient
            grad_norms = delta.grad.view(batch_size, -1).norm(p=2, dim=1)
            delta.grad.div_(grad_norms.view(-1, 1, 1, 1))
            # avoid nan or inf if gradient is 0
            if (grad_norms == 0).any():
                delta.grad[grad_norms == 0] = torch.randn_like(delta.grad[grad_norms == 0])
            optimizer_delta.step()

            # projection
            delta.data.add_(x_natural)
            delta.data.clamp_(0, 1).sub_(x_natural)
            delta.data.renorm_(p=2, dim=0, maxnorm=epsilon)
        x_adv = Variable(x_natural + delta, requires_grad=False)
    else:
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    #imsave(x_adv[0], "x_out/%d" % i, "pgd")
    return x_adv





def t_pgd(model, x_natural, target, step_size=0.003,
                epsilon=8.0/255.0, perturb_steps=10, distance='l_inf'):
    #imsave(x_natural[0], "x_out/%d" % i, "clean")
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).detach().cuda()

    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                logits = model(x_adv)
                loss_kl = F.cross_entropy(logits, target)

            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            x_adv = x_adv.detach() - step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    elif distance == 'l_2':
        batch_size = len(x_natural)
        delta = 0.001 * torch.randn(x_natural.shape).cuda().detach()
        delta = Variable(delta.data, requires_grad=True)

        # Setup optimizers
        optimizer_delta = optim.SGD([delta], lr=epsilon / perturb_steps * 2)

        for _ in range(perturb_steps):
            x_adv = x_natural - delta

            # optimize
            optimizer_delta.zero_grad()
            with torch.enable_grad():
                logits = model(x_adv)
                # logits = F.softmax(logits, dim=1)
                loss = F.cross_entropy(logits, target)
            loss.backward()
            # renorming gradient
            grad_norms = delta.grad.view(batch_size, -1).norm(p=2, dim=1)
            delta.grad.div_(grad_norms.view(-1, 1, 1, 1))
            # avoid nan or inf if gradient is 0
            if (grad_norms == 0).any():
                delta.grad[grad_norms == 0] = torch.randn_like(delta.grad[grad_norms == 0])
            optimizer_delta.step()

            # projection
            delta.data.add_(x_natural)
            delta.data.clamp_(0, 1).sub_(x_natural)
            delta.data.renorm_(p=2, dim=0, maxnorm=epsilon)
        x_adv = Variable(x_natural + delta, requires_grad=False)
    else:
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    #imsave(x_adv[0], "x_out/%d" % i, "pgd")
    return x_adv




#对齐
def lalign(x, y, alpha=2):
    return (x - y).norm(dim=1).pow(alpha)

#均匀
def lunif(x, t=2):
    sq_pdist = torch.pdist(x, p=2).pow(2)
    return sq_pdist.mul(-t).exp().mean()




def main():
    # init model, ResNet18() can be also used here for training
    setup_seed(args.seed)

    # setup data loader
    trans_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

    trans_test = transforms.Compose([
        transforms.ToTensor()
    ])


    trainset = data_dataset(img_path=args.nat_img_train, clean_label_path=args.nat_label_train,
                            transform=trans_train)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, drop_last=False,
                                               shuffle=True, num_workers=4, pin_memory=True)
    testset = data_dataset(img_path=args.nat_img_test, clean_label_path=args.nat_label_test, transform=trans_test)

    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, drop_last=False, shuffle=True,
                                              num_workers=4, pin_memory=True)




    # model = wideresnet.WideResNet(depth=34, num_classes=10, widen_factor=10).cuda()
    # model = wggnet.VGGNet19().to(device)
    # model = wideresnet.WideResNet(depth=28, num_classes=10, widen_factor=10, dropRate=0.3).to(device)
    for model_name in ['model_120', 'best_adv_model']:
        model = resnet.ResNet18(10).cuda()
        model_d = model_dir.split('/')[-1] + '_' + model_name

        if 'best' in model_name:
            model = torch.nn.DataParallel(model).cuda()
            model.module.load_state_dict(torch.load(os.path.join(model_dir, model_name + '.pth')))
        else:
            model.load_state_dict(torch.load(os.path.join(model_dir, model_name + '.pth')))
            model = torch.nn.DataParallel(model).cuda()

        if os.path.exists('./pictures_new/%s' % (model_d)) == False:
            os.mkdir('./pictures_new/%s' % (model_d))


        model.eval()

        with torch.no_grad():
            al_adv, dis_adv, al_nat, dis_nat = lode_feature_align(test_loader, model)
            un1, un2, un3 = lode_feature_uniform(test_loader, model)

        plt.title('TEST al_adv:%f, al_nat:%f' % (al_adv, al_nat))
        logger.info('TEST al_adv:%f, al_nat:%f, un_all:%f, un_nat:%f, un_adv:%f' % (al_adv, al_nat, un1, un2, un3))
        plt.hist(dis_adv, histtype='stepfilled', alpha=0.3, bins=100)
        plt.savefig('./pictures_new/%s/test_nat_adv_distance.png' % (model_d))
        plt.show()
        plt.clf()

        plt.hist(dis_nat, histtype='stepfilled', alpha=0.3, bins=100)
        plt.savefig('./pictures_new/%s/test_nat_nat_distance.png' % (model_d))
        plt.show()
        plt.clf()








if __name__ == '__main__':
    main()
