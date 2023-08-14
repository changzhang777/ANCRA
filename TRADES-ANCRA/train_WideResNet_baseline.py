from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
from torchvision import datasets, transforms
from models import resnet, wideresnet, preactresnet
from torch.autograd import Variable
import zipfile
import random
import numpy as np
import copy
import itertools
from tiny_imagenet import TinyImageNet
import logging

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description='PyTorch CIFAR TRADES Adversarial Training')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
                    help='input batch size for testing (default: 128)')
parser.add_argument('--epochs', type=int, default=120, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--weight-decay', '--wd', default=2e-4,
                    type=float, metavar='W')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--epsilon', default=8. / 255.,
                    help='perturbation')
parser.add_argument('--num-steps', default=10,
                    help='perturb number of steps')
parser.add_argument('--step-size', default=0.007,
                    help='perturb step size')
parser.add_argument('--beta', default=6.0, type=float,
                    help='regularization, i.e., 1/lambda in TRADES')
parser.add_argument('--alpha', default=1.0, type=float,
                    help='weight for neg to anchor decay')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')

parser.add_argument('--model-dir', default='/model/TRADES-WideResNet',
                    help='directory of model for saving checkpoint')

parser.add_argument('--save-freq', '-s', default=10, type=int, metavar='N',
                    help='save frequency')
parser.add_argument('--cr_beta', default=2.0, type=float,
                    help='regularization, i.e., beta in channel regularization')
parser.add_argument('--zeta', default=3.0, type=float,
                    help='for neg loss')
parser.add_argument('--sample_class_number', type=int, default=1, metavar='N',
                    help='how many neg classes are there when sampling for one anchor sample')
parser.add_argument('--sample_number', type=int, default=1,
                    help='how many neg samples are there when sampling in one neg class')
parser.add_argument('--strategy', type=str, default='random_adv', choices=['random', 'soft',
                                                                           'hard', 'easy', 'random_adv', 'soft_adv',
                                                                           'hard_adv', 'easy_adv'],
                    help='some strategy to pick up negivative samples')
parser.add_argument('--restart_epoch', type=int, default=1, metavar='N',
                    help='number of epochs to retrain')
parser.add_argument('--ckpt_url', default="/", help='pretrain model path')

args = parser.parse_args()

model_dir = args.model_dir + '-strategy-' + args.strategy + '-alpha-' + str(args.alpha) + '-beta-' + str(
    args.beta) + '-zeta-' + str(args.zeta) + '-seed-' + str(args.seed)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

logger = logging.getLogger(__name__)
logfile = os.path.join(model_dir, 'train.log')
logging.basicConfig(
    format='[%(asctime)s] - %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S',
    level=logging.INFO,
    filename=os.path.join(model_dir, 'train.log'))
logger.info(args)

seed = args.seed
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

sn = args.sample_number
scn = args.sample_class_number
alpha = args.alpha

class_number = 10  # len(trainset.classes)


def u_trades(model, x_natural, y, step_size=0.003,
             epsilon=8.0 / 255.0, perturb_steps=10, beta=1.0, distance='l_inf', cr_beta=2.0):
    # model.eval()
    criterion_kl = nn.KLDivLoss(size_average=False)

    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                output_adv, extraoutput_adv = model(x_adv, y, _eval=True)
                output_nat, extraoutput_nat = model(x_natural, y, _eval=True)
                loss_kl = criterion_kl(F.log_softmax(output_adv, dim=1),
                                       F.softmax(output_nat, dim=1))
                channel_reg_loss = 0.
                for i in range(len(extraoutput_adv)):
                    channel_reg_loss += criterion_kl(F.log_softmax(extraoutput_adv[i], dim=1),
                                                     F.softmax(extraoutput_nat[i], dim=1))
                channel_reg_loss /= len(extraoutput_adv)
                loss_kl += cr_beta * channel_reg_loss
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    else:
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    return x_adv


def u_pgd(model, x_natural, y, step_size=0.003,
          epsilon=8.0 / 255.0, perturb_steps=10, distance='l_inf', cr_beta=2.0):
    # model.eval()
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                output_adv = model(x_adv)
                loss_kl = F.cross_entropy(output_adv, y)

            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    else:
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    return x_adv


def t_pgd(model, x_natural, y, target, step_size=0.003,
          epsilon=8.0 / 255.0, perturb_steps=10, distance='l_inf', cr_beta=2.0):
    # model.eval()
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                output_adv, extra_output_adv = model(x_adv, y, _eval=True)
                loss_kl = F.cross_entropy(output_adv, target)
                channel_reg_loss = 0.
                for i in range(len(extra_output_adv)):
                    channel_reg_loss += F.cross_entropy(extra_output_adv[i], target)
                channel_reg_loss /= len(extra_output_adv)
                loss_kl += cr_beta * channel_reg_loss
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            x_adv = x_adv.detach() - step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    else:
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    return x_adv


def loss_fn_ori(logit1, logit2):
    value1, indice1 = torch.max(logit1, dim=-1)
    value2, indice2 = torch.max(logit2, dim=-1)
    indices = torch.nonzero(indice1 == indice2)
    loss = torch.Tensor([0]).squeeze().cuda()
    for index in indices:
        idx = index[0]
        loss += (value1[idx] * value2[idx]) ** 0.5
    return loss


def loss_fn(logit1, logit2):
    return alpha * loss_fn_ori(logit1.detach(), logit2) + (1 - alpha) * loss_fn_ori(logit2.detach(), logit1)


def trades_loss_with_neg(model,
                         x_natural,
                         y,
                         # data_neg,
                         # neg_gt,
                         optimizer,
                         beta=6.0,
                         zeta=args.zeta,
                         cr_beta=1.0):
    batch_size = len(x_natural)
    criterion_kl = nn.KLDivLoss(reduction='sum')
    x_adv = u_pgd(model=model, x_natural=x_natural, y=y, step_size=args.step_size,
                  epsilon=args.epsilon, perturb_steps=args.num_steps)
    model.train()
    """print('after upgd:')
    print(model.module.bn1.running_mean, model.module.bn1.running_var)"""
    # zero gradient
    optimizer.zero_grad()
    # calculate robust loss
    logits = model(x_natural)
    loss_natural = F.cross_entropy(logits, y)

    output_adv = model(x_adv)
    loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(output_adv, dim=1),
                                                    F.softmax(logits, dim=1))

    loss_robust_t = loss_robust.detach()
    # print(loss_natural, loss_robust, loss_robust_t)

    loss = loss_natural + beta * loss_robust  # + zeta * loss_robust_t
    """logits_eval, _ = model(x_natural, y, _eval=True)
    output_adv_eval, _ = model(x_adv, y, _eval=True)
    output_t_eval, _ = model(data_neg, y, _eval=True)"""
    return loss, loss_natural.detach(), loss_robust.detach(), loss_robust_t.detach()  # logits_eval.detach(), output_adv_eval.detach(), output_t_eval.detach()


def pick_neg(args, model, data, label, epoch):
    class_to_id = {i: [] for i in range(class_number)}
    label_neg = label.repeat(scn * sn, 1).t().flatten().cuda()
    for index, y in enumerate(label):
        class_to_id[y.item()].append(index)

    if 'random' in args.strategy or epoch <= 10:
        neg_class_l = [
            random.sample([num for num in range(class_number) if (num != lb) and (len(class_to_id[num]) >= sn)], scn)
            for lb in label]
        neg_class_l = list(itertools.chain.from_iterable(neg_class_l))
        neg_id = [random.sample(class_to_id[neg_cl], sn) for neg_cl in neg_class_l]
        neg_id = list(itertools.chain.from_iterable(neg_id))
    else:
        logits, _ = model(data, _eval=True)
        soft_label = torch.argmax(logits, -1)
        neg_id = []

        for i in range(soft_label.shape[0]):
            m = set()
            for j in range(soft_label.shape[0]):
                if 'soft' in args.strategy:
                    # 预测类别一致
                    if (soft_label[j] == soft_label[i]):
                        m.add(j)
                elif 'hard' in args.strategy:
                    # 预测类别和目标类一致
                    if (soft_label[j] == label[i]):
                        m.add(j)
                else:
                    # 预测类别和目标类别不一致
                    if (soft_label[j] != label[i]) and (soft_label[j] != soft_label[i]):
                        m.add(j)
            n = set(class_to_id[label[i].item()])
            neg_id_list = list(m - n)  # append
            if neg_id_list == []:
                neg_class_l = [random.sample(
                    [num for num in range(class_number) if (num != label[i].item()) and (len(class_to_id[num]) >= sn)],
                    scn)]
                neg_class_l = list(itertools.chain.from_iterable(neg_class_l))
                neg_idx = [random.sample(class_to_id[neg_cl], sn) for neg_cl in neg_class_l]
                neg_idx = list(itertools.chain.from_iterable(neg_idx))
                neg_id.extend(neg_idx)
                continue
            logits_neg = logits[neg_id_list]
            score = logits_neg[:, label[i].item()]
            _, nid = torch.topk(score, k=scn * sn)
            neg_id.extend([neg_id_list[id] for id in nid.cpu().numpy().tolist()])

    neg_gt = label[neg_id].cuda()

    if 'adv' in args.strategy:
        """print('before tpgd:')
        print(model.module.bn1.running_mean, model.module.bn1.running_var)"""
        data_neg = t_pgd(model=model, x_natural=data[neg_id].cuda(), y=neg_gt, target=label_neg,
                         step_size=args.step_size, epsilon=args.epsilon,
                         perturb_steps=args.num_steps, cr_beta=args.cr_beta)
        """print('after tpgd:')
        print(model.module.bn1.running_mean, model.module.bn1.running_var)"""
    else:
        data_neg = data[neg_id].cuda()
    return data_neg, neg_gt


def trades_train(args, model, train_loader, optimizer, epoch):
    model.train()
    Loss = 0.
    """
    Ap_l2 = 0.
    An_l2 = 0.
    Nat_correct = 0.
    Adv_correct_u = 0.
    Adv_correct_t = 0.
    """
    Class_loss = 0.
    Adv_loss = 0.
    Adv_t_loss = 0.
    for batch_idx, (data, label) in enumerate(train_loader):

        data, target = data.cuda(), label.cuda()
        # data_neg, neg_gt = pick_neg(args, model, data, target, epoch)

        optimizer.zero_grad()
        # calculate robust loss
        # , logits, output_adv, output_t = \
        loss, loss_natural, loss_robust, loss_robust_t = \
            trades_loss_with_neg(model=model,
                                 x_natural=data,
                                 y=target,
                                 # data_neg=data_neg,
                                 # neg_gt=neg_gt,
                                 optimizer=optimizer,
                                 beta=args.beta,
                                 zeta=args.zeta,
                                 cr_beta=args.cr_beta)
        loss.backward()
        optimizer.step()

        """nat_correct = (logits.data.max(1)[1] == target.data).float().mean()
        adv_correct_u = (output_adv.data.max(1)[1] == target.data).float().mean()
        adv_correct_t = (output_t.data.max(1)[1] == neg_gt.data).float().mean()

        ap_l2 = torch.linalg.norm(logits - output_adv).float() / (target.shape[0] ** 0.5)
        an_l2 = torch.linalg.norm(
            logits.unsqueeze(1).expand(target.shape[0], scn * sn, class_number).resize(scn * sn * target.shape[0], class_number)
            - output_t).float() / ((scn * sn * target.shape[0]) ** 0.5)"""

        # print progress
        if batch_idx % args.log_interval == 0:
            print(
                'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, class_loss: {:.4f}, adv_loss: {:.4f}, adv_t_loss: {:.4f}'.format(
                    # , Nat_Correct: {:.4f}, Adv_Correct_Untarget: {:.4f}, Adv_Correct_target: {:.4f}, ap_l2: {:.2f}, an_l2: {:.2f}
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item(), loss_natural, loss_robust,
                    loss_robust_t))  # ,
            # 100 * nat_correct, 100 * adv_correct_u, 100 * adv_correct_t, ap_l2, an_l2))
            logger.info(
                'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, class_loss: {:.4f}, adv_loss: {:.4f}, adv_t_loss: {:.4f}'.format(
                    # , Nat_Correct: {:.4f}, Adv_Correct_Untarget: {:.4f}, Adv_Correct_target: {:.4f}, ap_l2: {:.2f}, an_l2: {:.2f}
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item(), loss_natural, loss_robust,
                    loss_robust_t))  # ,
            # 100 * nat_correct, 100 * adv_correct_u, 100 * adv_correct_t, ap_l2, an_l2))

        num_batch = len(train_loader)
        """Adv_correct_u += 100 * adv_correct_u.item() / num_batch
        Adv_correct_t += 100 * adv_correct_t.item() / num_batch
        Nat_correct += 100 * nat_correct.item() / num_batch
        Ap_l2 += ap_l2.item() / num_batch
        An_l2 += an_l2.item() / num_batch"""
        Loss += loss.item() / num_batch
        Class_loss += loss_natural.item() / num_batch
        Adv_loss += loss_robust.item() / num_batch
        Adv_t_loss += loss_robust_t.item() / num_batch
    return Loss, Class_loss, Adv_loss, Adv_t_loss  # , Nat_correct, Adv_correct_u, Adv_correct_t, Ap_l2, An_l2


def eval_test(model, test_loader):
    model.eval()
    test_loss = 0
    test_loss_adv = 0
    correct = 0
    correct_adv = 0.
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            output = model(data)

            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
            test_loss += F.cross_entropy(output, target, size_average=False).item()

            adv_data = _pgd_whitebox(model, data, target, epsilon=8. / 255., num_steps=40, step_size=0.007)
            output_adv = model(adv_data)
            pred_adv = output_adv.max(1, keepdim=True)[1]
            correct_adv += pred_adv.eq(target.view_as(pred_adv)).sum().item()
            test_loss_adv += F.cross_entropy(output_adv, target, size_average=False).item()

    """print('after test:')
    print(model.module.bn1.running_mean, model.module.bn1.running_var)"""

    test_loss /= len(test_loader.dataset)
    test_loss_adv /= len(test_loader.dataset)
    print(
        'Test: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%, Average loss_adv: {:.4f}, Accuracy_adv: {}/{} ({:.0f}%)'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset), test_loss_adv, correct_adv, len(test_loader.dataset),
            100. * correct_adv / len(test_loader.dataset)))
    logger.info(
        'Test: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%, Average loss_adv: {:.4f}, Accuracy_adv: {}/{} ({:.0f}%)'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset), test_loss_adv, correct_adv, len(test_loader.dataset),
            100. * correct_adv / len(test_loader.dataset)))
    test_accuracy = correct / len(test_loader.dataset)
    test_accuracy_adv = correct_adv / len(test_loader.dataset)
    return test_loss, test_accuracy, test_accuracy_adv


def _pgd_whitebox(model,
                  X,
                  y,
                  epsilon=args.epsilon,
                  num_steps=args.num_steps,
                  step_size=args.step_size):
    X_pgd = Variable(X.data, requires_grad=True)

    random_noise = 0.001 * torch.randn(X.shape).cuda().detach()
    X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

    for _ in range(num_steps):
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()

        with torch.enable_grad():
            output = model(X_pgd)
            loss = nn.CrossEntropyLoss()(output, y)

        loss.backward()
        eta = step_size * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)
    return X_pgd


def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate"""
    lr = args.lr
    if epoch >= 75:
        lr = args.lr * 0.1
    if epoch >= 90:
        lr = args.lr * 0.01
    if epoch >= 100:
        lr = args.lr * 0.001
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main():
    # init model, ResNet18() can be also used here for training
    # setup data loader
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    trainset = torchvision.datasets.CIFAR10(root='./dataset', train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, **kwargs)
    testset = torchvision.datasets.CIFAR10(root='./dataset', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)

    model = wideresnet.WideResNet().cuda()
    if args.restart_epoch != 1:
        model.load_state_dict(torch.load(args.ckpt_url))
    model = nn.DataParallel(model).cuda()

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    best_adv_acc = 0.
    best_nat_acc = 0.
    best_epoch = 0

    for epoch in range(args.restart_epoch, args.epochs + 1):
        # adjust learning rate for SGD
        adjust_learning_rate(optimizer, epoch)

        # adversarial training
        # , Nat_correct, Adv_correct_u, Adv_correct_t, Ap_l2, An_l2 = \
        Loss, Class_loss, Adv_loss, Adv_t_loss = trades_train(args, model, train_loader, optimizer, epoch)
        """print('after train:')
        print(model.module.bn1.running_mean, model.module.bn1.running_var)"""

        # evaluation on natural examples
        print(
            'Train Epoch: {}, Loss: {:.6f}, NAT_Loss: {:.6f}, ADV_Loss: {:.6f}, ADV_T_Loss: {:.6f}'.format(
                # , Nat_Correct: {:.4f}, Adv_Correct_Untarget: {:.4f}, Adv_Correct_target: {:.4f}, AP_l2: {:.6f}, AN_l2: {:.6f}'.format(
                epoch, Loss, Class_loss, Adv_loss,
                Adv_t_loss))  # , Nat_correct, Adv_correct_u, Adv_correct_t, Ap_l2, An_l2))

        logger.info(
            'Train Epoch: {}, Loss: {:.6f}, NAT_Loss: {:.6f}, ADV_Loss: {:.6f}, ADV_T_Loss: {:.6f}'.format(
                # , Nat_Correct: {:.4f}, Adv_Correct_Untarget: {:.4f}, Adv_Correct_target: {:.4f}, AP_l2: {:.6f}, AN_l2: {:.6f}'.format(
                epoch, Loss, Class_loss, Adv_loss,
                Adv_t_loss))  # , Nat_correct, Adv_correct_u, Adv_correct_t, Ap_l2, An_l2))

        print('================================================================')
        logger.info('\n================================================================\n')

        _, nat_acc, adv_acc = eval_test(model, test_loader)
        print('================================================================')
        logger.info('\n================================================================\n')

        # save checkpoint
        save_state = {'model': model.module.state_dict(),
                      'optimizer': optimizer.state_dict()}
        torch.save(save_state, os.path.join(model_dir, 'last_state.pt'))

        if epoch % args.save_freq == 0:
            torch.save(model.module.state_dict(),
                       os.path.join(model_dir, 'model-%d.pt' % epoch))
        if adv_acc > best_adv_acc:
            torch.save(model.module.state_dict(),
                       os.path.join(model_dir, 'best_adv_model.pt'))
            best_adv_acc = adv_acc
            best_nat_acc = nat_acc
            best_epoch = epoch
            print(
                'Best Adv Epoch: {}, Nat_Correct: {:.4f}, Adv_Correct_Untarget: {:.4f}'.format(
                    best_epoch, best_nat_acc, best_adv_acc))
            logger.info(
                'Best Adv Epoch: {}, Nat_Correct: {:.4f}, Adv_Correct_Untarget: {:.4f}'.format(
                    best_epoch, best_nat_acc, best_adv_acc))
            print('================================================================')
            logger.info('\n================================================================\n')


if __name__ == '__main__':
    main()
