from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import torch.backends.cudnn as cudnn
from autoattack import AutoAttack
from data import data_dataset# , data_noise_dataset, distilled_dataset
#from models import resnet_transition
from models import resnet
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
parser.add_argument('--batch-size', type=int, default=1250, metavar='N',
                    help='input batch size for training (default: 500)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--dataset', type=str,
                    help='cifar10,cifar100,tiny-imagenet', default='cifar10')
parser.add_argument('--model-dir', default='./checkpoint/encoder/Trades_train-adv_loss-6.0-neg_avg_adv_loss-detach-nat-13.0-scn-1-sn-1_hy_MART',
                    help='directory of model for saving checkpoint')
parser.add_argument('--log_path', type=str, default='./logs')
parser.add_argument('--num-steps', default=40,
                    help='perturb number of steps')
parser.add_argument('--step-size', default=0.007,
                    help='perturb step size')


args = parser.parse_args()

logger = logging.getLogger(__name__)
logfile = os.path.join(args.model_dir, 'test.log')
logging.basicConfig(
    format='[%(asctime)s] - %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S',
    level=logging.INFO,
    filename=os.path.join(args.model_dir, 'test.log'))
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





def test_adv(model, data_adv, label):
    logits = model(data_adv)
    correct = (logits.data.max(1)[1] == label.data).float().sum().item()
    acc_adv_40 = 100 * correct / label.size(0)
    return acc_adv_40




def main():
    # init model, ResNet18() can be also used here for training
    setup_seed(args.seed)

    # setup data loader
    trans_test = transforms.Compose([
        transforms.ToTensor()
    ])

    #testset = data_dataset(img_path=args.nat_img_test, clean_label_path=args.nat_label_test, transform=trans_test)
    #test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, drop_last=False, shuffle=False,
    #                                          num_workers=4, pin_memory=True)
    testset = torchvision.datasets.CIFAR10(root='../../data', train=False, download=True, transform=trans_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, drop_last=False,
                                              num_workers=2, pin_memory=True)
    Logit_nat = []
    Logit_adv = []
    Epoch = [10 * i for i in range(1, 13)]
    for epoch in Epoch:
        path = 'model_%s.pth' % str(epoch)
        model_path = os.path.join(args.model_dir, path)
        print(model_path)
        logger.info(model_path)

        model = resnet.ResNet18(10).cuda()
        model.load_state_dict(torch.load(model_path))
        model = torch.nn.DataParallel(model).cuda()
        cudnn.benchmark = True
        model.eval()
        logit_nat, logit_adv = torch.zeros((10,)), torch.zeros((10,))
        for batch, (data, label) in enumerate(test_loader):
            data, label = data.to(device), label.to(device)
            index = torch.nonzero(label == 0)
            data = data[index].squeeze(1)
            #print(data.shape)
            label = torch.zeros(data.shape[0]).long().cuda()
            with torch.no_grad():
                logit_nat += F.softmax(model(data), dim=-1).mean(dim=0).cpu()

            data_adv = pgd_attack(model=model, x_natural=data, y=label, step_size=args.step_size,
                                  epsilon=args.epsilon, perturb_steps=args.num_steps)
            with torch.no_grad():
                logit_adv += F.softmax(model(data_adv), dim=-1).mean(dim=0).cpu()
        Logit_nat.append(logit_nat / len(test_loader))
        Logit_adv.append(logit_adv / len(test_loader))

    model_path = args.model_dir.split('/')[-1]
    x_width = [20 * i for i in range(10)]
    label = list(range(10))
    for i in range(len(Logit_nat)):
        plt.bar(x_width, Logit_nat[i], width=1.6, label=Epoch[i])
        plt.title('logits of natural example')
        plt.xlabel("class")
        plt.ylabel("logit")
        x_width = [i + 1.6 for i in x_width]
    plt.xticks([20 * i for i in range(10)], label)
    plt.legend()
    plt.savefig('./picture_hist/%s-logits of natural example.png' % (model_path))
    plt.show()
    plt.clf()

    x_width = [20 * i for i in range(10)]
    label = list(range(10))
    for i in range(len(Logit_adv)):
        plt.bar(x_width, Logit_adv[i], width=1.6, label=Epoch[i])
        plt.title('logits of adversarial example')
        plt.xlabel("class")
        plt.ylabel("logit")
        x_width = [i + 1.6 for i in x_width]
    plt.xticks([20 * i for i in range(10)], label)
    plt.legend()
    plt.savefig('./picture_hist/%s-logits of adversarial example.png' % (model_path))
    plt.show()
    plt.clf()


if __name__ == '__main__':
    main()

""" """
