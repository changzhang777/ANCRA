from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import torch.backends.cudnn as cudnn

import resnet_self_two_layer as resnet
import numpy as np
from torch.autograd import Variable
import torch.optim as optim
import logging
import cv2
import torchvision
import warnings
import matplotlib.pyplot as plt


warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='PyTorch CIFAR TRADES Adversarial Training')
parser.add_argument('--epsilon', type=float, default=8.0/255.0, help='perturb scale')
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 500)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--dataset', type=str,
                    help='cifar10,cifar100,tiny-imagenet', default='cifar10')
parser.add_argument('--model-dir', default='./model/NPsfc-cifar10-strategy-random-alpha-1.0-beta-6.0-zeta-3.0-seed-1',
                    help='directory of model for saving checkpoint')
parser.add_argument('--strategy', default='random')
parser.add_argument('--log_path', type=str, default='./logs')
parser.add_argument('--num-steps', default=10,
                    help='perturb number of steps')
parser.add_argument('--step-size', default=0.007,
                    help='perturb step size')
parser.add_argument('--target_label', type=int, default=0)

args = parser.parse_args()

logger = logging.getLogger(__name__)
logfile = os.path.join(args.model_dir, 'eval.log')
logging.basicConfig(
    format='[%(asctime)s] - %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S',
    level=logging.INFO,
    filename=os.path.join(args.model_dir, 'eval.log'))
logger.info(args)

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)




def eval_test(model, device, test_loader):
    model.eval()
    test_loss = 0.
    correct = 0.

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            output = model(data)

            test_loss += F.cross_entropy(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]

            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('Test: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    logger.info('Test: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    test_accuracy = correct / len(test_loader.dataset)
    return test_loss, test_accuracy





def imsave(tensor, name):
    toPIL = transforms.ToPILImage()  # 这个函数可以将张量转为PIL图片，由小数转为0-255之间的像素值
    image = tensor.clone() # we clone the tensor to not do changes on it
    image = image.squeeze(0)  # remove the fake batch dimension
    #print(image.shape, image.type)
    image = toPIL(image.float())

    image.save('./pictures/{}.jpg'.format(name))
    np.savetxt('./numbers/{}.csv'.format(name), tensor.view(tensor.size(0), -1).cpu().numpy())
    for i in range(3):
        np.savetxt('./numbers/{}_{}.csv'.format(name, i+1), tensor.cpu().numpy()[i])



def loss_fn(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)







def pgd_attack(model, x_natural, y, step_size=0.003,
                epsilon=8.0/255.0, perturb_steps=10, distance='l_inf', i=0):
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
    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    return x_adv




def t_pgd(model, x_natural, y, target, step_size=0.003,
          epsilon=8.0 / 255.0, perturb_steps=10, distance='l_inf', cr_beta=2.0):
    model.eval()
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
    # model.train()
    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    return x_adv




import random
import itertools

def pick_neg(args, model, data, label, data_all, target):
    class_to_id = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []}
    scn = sn = 1
    label_neg = label.clone().cuda()
    for index, y in enumerate(target):
        class_to_id[y.item()].append(index)

    if 'random' in args.strategy:
        neg_class_l = [random.sample([num for num in range(10) if (num != lb) and (len(class_to_id[num]) >= sn)], scn)
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
                    [num for num in range(10) if (num != label[i].item()) and (len(class_to_id[num]) >= 1)], 1)]
                neg_class_l = list(itertools.chain.from_iterable(neg_class_l))
                neg_idx = [random.sample(class_to_id[neg_cl], 1) for neg_cl in neg_class_l]
                neg_idx = list(itertools.chain.from_iterable(neg_idx))
                neg_id.extend(neg_idx)
                continue
            logits_neg = logits[neg_id_list]
            score = logits_neg[:, label[i].item()]
            _, nid = torch.topk(score, k=1)
            neg_id.extend([neg_id_list[id] for id in nid.cpu().numpy().tolist()])

    neg_gt = target[neg_id].cuda()

    if 'adv' in args.strategy:
        data_neg = t_pgd(model=model, x_natural=data_all[neg_id].cuda(), y=neg_gt, target=label_neg,
                         step_size=args.step_size, epsilon=args.epsilon,
                         perturb_steps=args.num_steps, cr_beta=2.0)

    else:
        data_neg = data_all[neg_id].cuda()
    return data_neg, neg_gt






def test_adv(model, data_adv, label):
    logits = model(data_adv)
    correct = (logits.data.max(1)[1] == label.data).float().sum().item()
    acc_adv_40 = 100 * correct / label.size(0)
    return acc_adv_40




def main():
    # init model, ResNet18() can be also used here for training
    setup_seed(args.seed)

    trans_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

    trans_test = transforms.Compose([
        transforms.ToTensor()
    ])

    trainset = torchvision.datasets.CIFAR10(root='../../data', train=True, download=True, transform=trans_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, drop_last=False,
    num_workers = 4, pin_memory = True)
    testset = torchvision.datasets.CIFAR10(root='../../data', train=False, download=True, transform=trans_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, drop_last=False,
                                              num_workers=2, pin_memory=True)

    path = 'model-wrn-last.pt'
    model_path = os.path.join(args.model_dir, path)
    print(model_path)
    logger.info(model_path)

    print('neg chosen from natural examples')
    logger.info('neg chosen from natural examples')

    model = resnet.ResNet18_S(10).cuda()
    model.load_state_dict(torch.load(model_path))
    model = torch.nn.DataParallel(model).cuda()
    cudnn.benchmark = True
    model.eval()
    PPfcos, PPpcos = torch.tensor([]), torch.tensor([])
    PPfL2, PPpL2 = torch.tensor([]), torch.tensor([])
    NPfcos, NPpcos = torch.tensor([]), torch.tensor([])
    NPfL2, NPpL2 = torch.tensor([]), torch.tensor([])

    for batch, (data, label) in enumerate(test_loader):
        data, target = data.to(device), label.to(device)
        data_all = data.detach()
        index = torch.nonzero(label == args.target_label)
        data = data[index].squeeze(1)
        label = torch.zeros(data.shape[0]).long().cuda()
        out, f = model(x=data, prejection=True)
        out, f = out.detach(), f.detach()
        data_adv = pgd_attack(model=model, x_natural=data, y=label, step_size=args.step_size,
                              epsilon=args.epsilon, perturb_steps=args.num_steps)
        out_adv, f_adv = model(x=data_adv, prejection=True)
        out_adv, f_adv = out_adv.detach(), f_adv.detach()
        PPfcos = torch.cat((PPfcos, F.cosine_similarity(f, f_adv).cpu()))
        PPfL2 = torch.cat((PPfL2, torch.norm(f - f_adv, p=2, dim=-1).cpu()))
        """PPpcos = torch.cat((PPpcos, F.cosine_similarity(out, out_adv).cpu()))
        PPpL2 = torch.cat((PPpL2, torch.norm(out - out_adv, p=2, dim=-1).cpu()))"""

        data_adv, neg_gt = pick_neg(args, model, data, label, data_all, target)
        out_adv, f_adv = model(x=data_adv, prejection=True)
        out_adv, f_adv = out_adv.detach(), f_adv.detach()
        NPfcos = torch.cat((NPfcos, F.cosine_similarity(f, f_adv).cpu()))
        NPfL2 = torch.cat((NPfL2, torch.norm(f - f_adv, p=2, dim=-1).cpu()))
        """NPpcos = torch.cat((NPpcos, F.cosine_similarity(out, out_adv).cpu()))
        NPpL2 = torch.cat((NPpL2, torch.norm(out - out_adv, p=2, dim=-1).cpu()))"""

    #print(PPcos.shape)
    Maxc = torch.max(PPfcos).item()
    Minc = torch.min(PPfcos).item()
    Meanc = torch.mean(PPfcos).item()
    Varc = torch.var(PPfcos).item()
    print('PP-f-Cosine: Max: {}, Min: {}, Mean: {}, Var: {}'.format(Maxc, Minc, Meanc, Varc))
    logger.info('PP-f-Cosine: Max: {}, Min: {}, Mean: {}, Var: {}'.format(Maxc, Minc, Meanc, Varc))

    Maxl = torch.max(PPfL2).item()
    Minl = torch.min(PPfL2).item()
    Meanl = torch.mean(PPfL2).item()
    Varl = torch.var(PPfL2).item()
    print('PP-f-fL2: Max: {}, Min: {}, Mean: {}, Var: {}'.format(Maxl, Minl, Meanl, Varl))
    logger.info('PP-f-fL2: Max: {}, Min: {}, Mean: {}, Var: {}'.format(Maxl, Minl, Meanl, Varl))

    Maxc = torch.max(NPfcos).item()
    Minc = torch.min(NPfcos).item()
    Meanc = torch.mean(NPfcos).item()
    Varc = torch.var(NPfcos).item()
    print('NP-f-Cosine: Max: {}, Min: {}, Mean: {}, Var: {}'.format(Maxc, Minc, Meanc, Varc))
    logger.info('NP-f-Cosine: Max: {}, Min: {}, Mean: {}, Var: {}'.format(Maxc, Minc, Meanc, Varc))

    Maxl = torch.max(NPfL2).item()
    Minl = torch.min(NPfL2).item()
    Meanl = torch.mean(NPfL2).item()
    Varl = torch.var(NPfL2).item()
    print('NP-f-L2: Max: {}, Min: {}, Mean: {}, Var: {}'.format(Maxl, Minl, Meanl, Varl))
    logger.info('NP-f-L2: Max: {}, Min: {}, Mean: {}, Var: {}'.format(Maxl, Minl, Meanl, Varl))




    #p
    """Maxc = torch.max(PPpcos).item()
    Minc = torch.min(PPpcos).item()
    Meanc = torch.mean(PPpcos).item()
    Varc = torch.var(PPpcos).item()
    print(Maxc, Minc, Meanc, Varc)
    logger.info('PP-p-Cosine: Max: {}, Min: {}, Mean: {}, Var: {}'.format(Maxc, Minc, Meanc, Varc))

    Maxl = torch.max(PPpL2).item()
    Minl = torch.min(PPpL2).item()
    Meanl = torch.mean(PPpL2).item()
    Varl = torch.var(PPpL2).item()
    print(Maxl, Minl, Meanl, Varl)
    logger.info('PP-p-L2: Max: {}, Min: {}, Mean: {}, Var: {}'.format(Maxl, Minl, Meanl, Varl))

    Maxc = torch.max(NPpcos).item()
    Minc = torch.min(NPpcos).item()
    Meanc = torch.mean(NPpcos).item()
    Varc = torch.var(NPpcos).item()
    print(Maxc, Minc, Meanc, Varc)
    logger.info('NP-p-Cosine: Max: {}, Min: {}, Mean: {}, Var: {}'.format(Maxc, Minc, Meanc, Varc))

    Maxl = torch.max(NPpL2).item()
    Minl = torch.min(NPpL2).item()
    Meanl = torch.mean(NPpL2).item()
    Varl = torch.var(NPpL2).item()
    print(Maxl, Minl, Meanl, Varl)
    logger.info('NP-p-L2: Max: {}, Min: {}, Mean: {}, Var: {}'.format(Maxl, Minl, Meanl, Varl))"""

    """torch.save(PPpcos, './three/%d-ours-PPpcos.npy' % args.target_label)
    torch.save(PPpL2, './three/%d-ours-PPpL2.npy' % args.target_label)
    torch.save(NPpcos, './three/%d-ours-NPpcos.npy' % args.target_label)
    torch.save(NPpL2, './three/%d-ours-NPpL2.npy' % args.target_label)"""
    torch.save(PPfcos, './three/%d-ours-PPfcos.npy' % args.target_label)
    torch.save(PPfL2, './three/%d-ours-PPfL2.npy' % args.target_label)
    torch.save(NPfcos, './three/%d-ours-NPfcos.npy' % args.target_label)
    torch.save(NPfL2, './three/%d-ours-NPfL2.npy' % args.target_label)

    model_path = args.model_dir#.split('/')[-1]

    """plt.hist(PPfcos.detach().numpy(), label='f', alpha=0.5)
    plt.hist(PPpcos.detach().numpy(), label='p', alpha=0.5)
    plt.title('Cosine_smiilarity of PP')
    plt.xlabel("cosine")
    plt.ylabel("rate")
    plt.legend()
    plt.savefig('%s/N-Cosine_smiilarity of PP.png' % (model_path))
    plt.show()
    plt.clf()


    plt.hist(PPfL2.detach().numpy(), label='f', alpha=0.5)
    plt.hist(PPpL2.detach().numpy(), label='p', alpha=0.5)
    plt.title('L2_smiilarity of PP')
    plt.xlabel("L2")
    plt.ylabel("rate")
    plt.legend()
    plt.savefig('%s/N-L2_smiilarity of PP.png' % (model_path))
    plt.show()
    plt.clf()

    plt.hist(NPfcos.detach().numpy(), label='f', alpha=0.5)
    plt.hist(NPpcos.detach().numpy(), label='p', alpha=0.5)
    plt.title('Cosine_smiilarity of PP')
    plt.xlabel("cosine")
    plt.ylabel("rate")
    plt.legend()
    plt.savefig('%s/N-Cosine_smiilarity of NP.png' % (model_path))
    plt.show()
    plt.clf()

    plt.hist(NPfL2.detach().numpy(), label='f', alpha=0.5)
    plt.hist(NPpL2.detach().numpy(), label='p', alpha=0.5)
    plt.title('L2_smiilarity of PP')
    plt.xlabel("L2")
    plt.ylabel("rate")
    plt.legend()
    plt.savefig('%s/N-L2_smiilarity of NP.png' % (model_path))
    plt.show()
    plt.clf()"""

    #p




if __name__ == '__main__':
    main()

""" """
