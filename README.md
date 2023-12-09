# Enhancing Robust Representation in Adversarial Training: Alignment and Exclusion Criteria (ANCRA)

This repository is the official implementation of [Enhancing Robust Representation in Adversarial Training: Alignment and Exclusion Criteria](https://arxiv.org/abs/2310.03358), its former name is ``Robust Representation Learning via Asymmetric Negative Contrast and Reverse Attention''. 

## ANCRA

![ANCRA](https://github.com/changzhang777/ANCRA/blob/main/figure/overview.png)

We propose ANCRA, a generic AT framework to address the overlook of robust representation learning. We propose two characteristics of robust feature to guide AT: **Exclusion** and **Alignment**. Following Exclusion, ANC freezes natural examples to alleviate class confusion and only pushes away examples of other classes in the feature space. To improve Alignment, RA weights feature by the partial parameters of the linear classifier, to provide class information and align feature of the same class. ANCRA can be combined with other defenses in a plug-and-play manner. ANCRA achieves state-of-the-art performance on three benchmark datasets.

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Training
```train
# To train ANCRA models
python train_CIFAR10_ResNet.py
python train_CIFAR10_WideResNet.py
python train_Tiny_ImageNet_PreActResNet.py

# To train baselines:
python train_ResNet18_baseline.py
python train_WideResNet_baseline.py
python train_PreActResNet_baseline.py
```

## Evaluation
```test
# To test robust accuracy against white-box attacks
python test_white_attack.py --ckpt_url $checkpoint_path$

# To draw feature visualization
python feature_distance.py --model_dir $checkpoint_path$ --dataset $dataset_type$ --target_label $label_class$
python draw_umap.py --model_dir $checkpoint_path$ --dataset $dataset_type$
python draw_hist.py
```

## Performance

:o: denotes training with additional data.

### ResNet-18 CIFAR10

| Method         | Natural Acc  | AutoAttack Acc |
| ------------------ |---------------- | -------------- |
| [Sehwag et al. (2021)](https://arxiv.org/abs/2104.09425):o: |     87.35%         |      58.50%       |
| [Addepalli et al. (2022)](https://arxiv.org/abs/2210.15318)  |     85.71%         |      52.48%       |
| ANCRA-PGD    |     85.10%         |      59.15%       |
| ANCRA-TRADES |     81.70%         |      59.70%       |
| ANCRA-MART   |     84.88%         |      59.60%       |


### ResNet-18 CIFAR100

| Method       | Natural Acc  | AutoAttack Acc |
| ------------------ |---------------- | -------------- |
| [Addepalli et al. (2022)](https://arxiv.org/abs/2210.15318)  |     65.45%         |      27.67%       |
| ANCRA-PGD   |     59.73%         |      34.44%       |
| ANCRA-TRADES   |     53.73%         |      35.81%       |
| ANCRA-MART   |     60.10%         |      35.05%       |

### WideResNet CIFAR10

| Model  |     Method      | Natural Acc  | AutoAttack Acc |
| ------------------ |---------------- | -------------- |  -------------- |
| WRN-34-10 | [Rade et al. (2021)](https://openreview.net/forum?id=BuD2LmNaU3a)   |     91.47%         |      62.83%       |
| WRN-34-10 | [Bui et al. (2022)](https://arxiv.org/abs/2202.13437)  |     84.93%         |      54.45%       |
| WRN-34-10 | ANCRA-TRADES   |     83.19%         |      66.28%       |
| WRN-28-10 | [Cui et al. (2023)](https://arxiv.org/abs/2305.13948):o:   |     92.16%         |      67.73%       |
| WRN-28-10 | [Gowal et al. (2020)](https://arxiv.org/abs/2010.03593)   |     89.48%         |      62.76%       |
| WRN-28-10 | ANCRA-TRADES   |     83.61%         |      65.87%       |

### PreActResNet-18 Tiny-ImageNet

| Method         | Natural Acc  | PGD-40 Acc |
| ------------------ |---------------- | -------------- |
| [Zhang et al. (2022)](https://arxiv.org/abs/2203.06616)    |     45.26%         |      18.42%       |
| ANCRA-PGD    |     43.02%         |      29.79%       |
| ANCRA-TRADES |     38.94%         |      31.24%       |
| ANCRA-MART   |     43.83%         |      31.44%       |


## Citation

If any parts of our paper and code help your research, please consider citing us and giving a star to our repository.

```
@inproceedings{Zhou2023EnhancingRR,
  title={Enhancing Robust Representation in Adversarial Training: Alignment and Exclusion Criteria},
  author={Nuoyan Zhou and Nannan Wang and Decheng Liu and Dawei Zhou and Xinbo Gao},
  year={2023},
  url={https://api.semanticscholar.org/CorpusID:263672090}
}
```

## Contact

If you have any questions or concerns, feel free to open issues or directly contact me through the ways on my GitHub homepage **provide below paper's title**.
