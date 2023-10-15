from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import torch.backends.cudnn as cudnn

import numpy as np
from torch.autograd import Variable
import torch.optim as optim
import logging
import cv2
import torchvision
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

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
parser.add_argument('--log_path', type=str, default='./logs')
parser.add_argument('--target_label', type=int, default=0)



args = parser.parse_args()

logger = logging.getLogger(__name__)
logfile = './pictures/eval.log'
logging.basicConfig(
    format='[%(asctime)s] - %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S',
    level=logging.INFO,
    filename=logfile)
logger.info(args)

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")



def main():
    method = ['PGD-AT', 'Trades', 'Mart', 'ours']#, 'ours'
    sim = ['PPfcos', 'PPfL2', 'NPfcos', 'NPfL2']#'PPpcos', 'PPpL2', 'NPpcos', 'NPpL2',
    labels = ['PGD-AT', 'TRADES', 'MART', 'ANCRA']#'ours'
    colors = ['#2B8FB8', '#A5678E', '#38549C', '#E8B7D4']#, '#E8B7D4'

    for i in range(len(sim)):
        for j in range(len(method)):
            sim_list = torch.load('./three/%s-%s.npy' % (method[j], sim[i]))#args.target_label,
            Maxc = torch.max(sim_list).item()
            Minc = torch.min(sim_list).item()
            Meanc = torch.mean(sim_list).item()
            Varc = torch.var(sim_list).item()
            print('{}-{}-Cosine: Max: {}, Min: {}, Mean: {}, Var: {}'.format(sim[i], method[j], Maxc, Minc, Meanc, Varc))
            logger.info('{}-{}-Cosine: Max: {}, Min: {}, Mean: {}, Var: {}'.format(sim[i], method[j], Maxc, Minc, Meanc, Varc))
            plt.hist(sim_list.detach().numpy(), bins=20, density=True, label=labels[j], alpha=0.9, color=colors[j], lw=2)
            #sns.distplot(sim_list.detach().numpy(), hist=True, bins=20, kde=False, hist_kws={'color': colors[j]}, kde_kws={'color': colors[j], 'linestyle': '-'},
            #            label=method[j])
        ax = plt.gca()
        plt.tick_params(width=2, labelsize=15)
        ax.spines['bottom'].set_linewidth(2)
        ax.spines['left'].set_linewidth(2)
        ax.spines['right'].set_linewidth(2)
        ax.spines['top'].set_linewidth(2)

        label = ax.get_xticklabels() + ax.get_yticklabels()
        [l.set_fontsize(14) for l in label]
        #ax.set_xticklabels(ax.get_xticklabels(), fontsize=14, weight='bold')
        #ax.set_yticklabels(ax.get_yticklabels(), fontsize=14, weight='bold')
        if i % 2 == 0:
            plt.xlabel("Cosine similarity", fontsize=15,weight='bold')
        else:
            plt.xlabel("L2 distance", fontsize=15,weight='bold')
        plt.ylabel("Frequency", fontsize=15,weight='bold')
        plt.legend(fontsize=15)
        plt.savefig('./pictures/Class-0-four-%s.png' % (sim[i]))#args.target_label,
        plt.show()
        plt.clf()


if __name__ == '__main__':
    main()

