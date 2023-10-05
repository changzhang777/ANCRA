# Robust Representation Learning via Asymmetric Negative Contrast and Reverse Attention (ANCRA)

(1)You can directly run the code. We have kept the best settings of hyperparameters.   

(2) You can easily change the code for CIFAR10 to the code for CIFAR100 by changing the data loader.  

## For train ANCRA models:
```bash
  $ python train_CIFAR10_ResNet.py
  $ python train_CIFAR10_WideResNet.py
  $ python train_Tiny_ImageNet_PreActResNet.py
```

## For train baselines:
```bash
  $ python train_ResNet18_baseline.py
  $ python train_ResNet18_baseline.py
  $ python train_PreActResNet_baseline.py
```

## For test: 
```bash
  $ python test_white_attack.py --ckpt_url $checkpoint_path$
```

## For eval: 
```bash
  $ python feature_distance.py --model_dir $checkpoint_path$ --dataset $dataset_type$ --target_label $label_class$
  $ python draw_umap.py --model_dir $checkpoint_path$ --dataset $dataset_type$
  $ python draw_hist.py
```

## training setting:

* $\alpha$ : 1.0, 

* $\beta$ : 6.0 in TRADES and 5.0 in MART,

* $\zeta$ : 3.0 (6.0 when training WideResNet-28-10),

* learning rate : **0.1**, 

* epochs : 120 (76), 

* batch size : 128, 

* weight decay : **0.0002**, (**0.0005** when training PreActResNet-18)






