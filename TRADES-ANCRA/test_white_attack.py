from __future__ import print_function
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable
import torch.optim as optim
from torchvision import datasets, transforms
from models.resnet import *
from autoattack import AutoAttack
import logging

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
parser = argparse.ArgumentParser(description='PyTorch CIFAR PGD Attack Evaluation')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                    help='model architecture')
parser.add_argument('--test-batch-size', type=int, default=200,
                    help='input batch size for testing (default: 200)')
parser.add_argument('--epsilon', default=8. / 255., type=float,
                    help='perturbation')
parser.add_argument('--num-steps', default=40, type=int,
                    help='perturb number of steps')
parser.add_argument('--step-size', default=0.007, type=float,
                    help='perturb step size')
parser.add_argument('--random', default=True,
                    help='random initialization for PGD')
parser.add_argument('--ckpt_url', default='./checkpoint/baseline/Mart_128_resnet18_cifar100', type=str,
                    help='directory of model checkpoints for white-box attack evaluation')

args = parser.parse_args()
model_path = args.ckpt_url

logger = logging.getLogger(__name__)
logfile = os.path.join(model_path, 'test.log')
logging.basicConfig(
    format='[%(asctime)s] - %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S',
    level=logging.INFO,
    filename=os.path.join(model_path, 'test.log'))
logger.info(args)

# set up data loader
transform_test = transforms.Compose([transforms.ToTensor(), ])
testset = torchvision.datasets.CIFAR100(root='./dataset', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, num_workers=4,
                                          pin_memory=True)


def _pgd_whitebox(model, X, y, epsilon=args.epsilon, num_steps=args.num_steps, step_size=args.step_size):
    out = model(X)
    acc = (out.data.max(1)[1] == y.data).sum().item()

    X_pgd = Variable(X.data, requires_grad=True)
    if args.random:
        random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).cuda(non_blocking=True)
        X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

    for _ in range(num_steps):
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()

        with torch.enable_grad():
            loss = nn.CrossEntropyLoss()(model(X_pgd), y)
        loss.backward()
        eta = step_size * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)
    acc_pgd = (model(X_pgd).data.max(1)[1] == y.data).sum().item()
    return acc, acc_pgd


def eval_adv_test_whitebox(model, test_loader):
    robust_accs = 0.
    natural_accs = 0.
    model.eval()

    for data, target in test_loader:
        data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
        X, y = Variable(data, requires_grad=True), Variable(target)
        acc, acc_pgd = _pgd_whitebox(model, X, y)
        natural_accs += acc
        robust_accs += acc_pgd
        # print('natural acc: {:.4f}, robust acc: {:.4f}'.format(natural_accs.avg, robust_accs.avg))
    natural_accs = natural_accs / len(test_loader.dataset)
    robust_accs = robust_accs / len(test_loader.dataset)
    print('natural acc: {:.4f}, pgd40 robust acc: {:.4f}'.format(natural_accs, robust_accs))
    logger.info('natural acc: {:.4f}, pgd40 robust acc: {:.4f}'.format(natural_accs, robust_accs))


def eval_adv_test_whitebox_fgsm(model, test_loader):
    robust_accs = 0.
    model.eval()

    for data, target in test_loader:
        data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
        X, y = Variable(data, requires_grad=True), Variable(target)
        acc, acc_pgd = _pgd_whitebox(model, X, y, num_steps=1,
                                     step_size=8 / 255.0)
        robust_accs += acc_pgd

    robust_accs = robust_accs / len(test_loader.dataset)
    print('fgsm robust acc: {:.4f}'.format(robust_accs))
    logger.info('fgsm robust acc: {:.4f}'.format(robust_accs))


def _cw_whitebox(model,
                 X,
                 y,
                 epsilon=args.epsilon,
                 num_steps=20,
                 step_size=args.step_size):
    # out = model(X)
    # err = (out.data.max(1)[1] != y.data).float().sum()
    X_pgd = Variable(X.data, requires_grad=True)

    random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).cuda()
    X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

    for _ in range(num_steps):
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()

        with torch.enable_grad():
            output = model(X_pgd)
            correct_logit = torch.sum(torch.gather(output, 1, (y.unsqueeze(1)).long()).squeeze())
            tmp1 = torch.argsort(output, dim=1)[:, -2:]
            new_y = torch.where(tmp1[:, -1] == y, tmp1[:, -2], tmp1[:, -1])
            wrong_logit = torch.sum(torch.gather(output, 1, (new_y.unsqueeze(1)).long()).squeeze())
            loss = - F.relu(correct_logit - wrong_logit)

        loss.backward()
        eta = step_size * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)
    output = model(X_pgd)
    err_pgd = (output.data.max(1)[1] != y.data).sum().item()
    return err_pgd


def eval_adv_test_whitebox_cw(model, test_loader):
    model.eval()
    robust_err_total = 0
    for data, target in test_loader:
        data, target = data.cuda(), target.cuda()
        X, y = Variable(data, requires_grad=True), Variable(target)
        err_robust = _cw_whitebox(model, X, y)
        robust_err_total += err_robust
    print('cw robust_acc: %.4f\n' % (1 - robust_err_total / len(test_loader.dataset)))
    logger.info('cw robust_acc: %.4f\n' % (1 - robust_err_total / len(test_loader.dataset)))


def eval_adv_test_AA(model, test_loader):
    model.eval()
    robust_err_total = 0
    adversary = AutoAttack(model, norm="Linf", eps=args.epsilon, version='standard',
                           log_path=os.path.join(model_path, 'autoattack.log'))
    for data, target in test_loader:
        data, target = data.cuda(), target.cuda()
        data_adv = adversary.run_standard_evaluation(data, target, bs=len(data))
        err_robust = (model(data_adv).data.max(1)[1] != target.data).sum().item()
        robust_err_total += err_robust
    print('Autoattack robust_acc: %.4f\n' % (1 - robust_err_total / len(test_loader.dataset)))
    logger.info('Autoattack robust_acc: %.4f\n' % (1 - robust_err_total / len(test_loader.dataset)))


def main():
    model = ResNet18_RA(num_classes=10).cuda()# (num_classes=100).cuda()
    model_name = 'model_120.pth'
    print("evaluating {}...".format(model_path))
    logger.info("evaluating {}...".format(os.path.join(model_path, model_name)))
    model.load_state_dict(torch.load(os.path.join(model_path, model_name)))
    model = torch.nn.DataParallel(model).cuda()
    eval_adv_test_whitebox(model, test_loader)

    eval_adv_test_whitebox_fgsm(model, test_loader)
    eval_adv_test_whitebox_cw(model, test_loader)
    eval_adv_test_AA(model, test_loader)


if __name__ == '__main__':
    main()
