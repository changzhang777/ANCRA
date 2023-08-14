# A code for ANCRA
(1)You can directly run the code. We have kept the best settings of hyperparameters.   

(2) You can easily change the code for CIFAR10 to the code for CIFAR100 by changing the data loader.  

(3)There are some bugs in the code for WideResNet. Please neglect it.

# ResNet18/WideResNet28/WideResNet34:
We adopt the SGD optimizer with a learning rate of **0.01**, a momentum of 0.9, epochs of 120 and a batch size of 128 as MART. And we set the weight decay as **0.0002** as TRADES. For the trade-off hyperparameters $\beta$, we use 6.0 in TRADES and 5.0 in MART, following the original setting in their papers. For other hyperparameters, $\alpha$ equals 1.0 and $\zeta$ equals 3.0.

# PreActResNet18:
We adopt the SGD optimizer with a learning rate of **0.1**, a weight decay of **0.0005**, a momentum of 0.9, epochs of 120 and a batch size of 128 as TRADES.
