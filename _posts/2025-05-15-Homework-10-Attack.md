---
title: 李宏毅-ML2022-HW10-Attack
date: 2025-05-15 15:07:00 +/-8
categories: [Machine Learning 2022]
tags: [Hung-yi Lee]     # TAG names should always be lowercase
---

## Task Description

本次作业的主题是“**Adversarial Attack**”，即**对抗攻击**，是一种通过对输入数据添加精心设计的微小扰动，使机器学习模型产生错误预测的技术。这些扰动对人类几乎不可察觉，但能显著改变模型输出。

### Prerequisite

对抗攻击按照**攻击目标**可分为Targeted attack（有目标攻击）和Non-targeted attack（无目标攻击），作业实作**Non-targeted attack**。

- Targeted attack：误导模型输出特定错误类别；
- Non-targeted attack：仅需使模型输出错误；

扰动必须**限制**在**人类不可感知**的范围内，即原图像$x^0$与扰动后图像$x$之间的距离$d(x^0, x) \le \epsilon$，使用L-infinity计算$d(x^0, x)$，即：
$$
d(x^0, x) = \|\Delta x\|_\infty = \text max \{ |\Delta x_1|, |\Delta x_2|, |\Delta x_3|,...\}
$$
其中：$x - x^0 = \Delta x$。

作业使用的**攻击算法**为Fast Gradient Sign Method (FGSM)或Iterative FGSM（I-FGSM ），详情参见[上课录影](https://www.youtube.com/watch?v=xGQKhbjrFRk)。

根据**攻击模式**可分为White Box Attack（白盒攻击）和Black Box Attack（黑盒攻击），作业使用黑盒攻击。

- 白盒攻击：攻击者知道目标模型参数，直接求梯度生成attacked objects；
- 黑盒攻击：攻击者对模型没有了解，训练一个与目标模型相似的代理模型，成功攻击代理模型的attacked objects也许在目标模型上也可以成功。

## Data Format

**Images**: 

- CIFAR-10 images 
- (32 * 32 RGB images) * 200 
  - airplane/airplane1.png, …, airplane/airplane20.png 
  - … 
  - truck/truck1.png, …, truck/truck20.png 
- 10 classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck) 
- 20 images for each class

## Methodology

1. 选择任意代理模型攻击JudgeBoi上的黑盒模型；

2. 实现**non-targeted adversarial attack method**：

   a. FGSM 

   b. I-FGSM 

   c. MI-FGSM

3. 通过多样化输入（Diverse input, DIM）增加攻击的transferability；

4. Ensemble attack：攻击多个代理模型。

## 思路

### Simple baseline (acc <= 0.70)

运行Sample Code。

fgsm_acc = 0.59000, fgsm_loss = 2.49187

### Medium baseline (acc <= 0.50)
根据助教提示，使用Ensemble Attack方法，攻击算法为IFGSM。

精度明显降低：

ifgsm_ensemble_acc = 0.00000, ifgsm_ensemble_loss = 2.45724

- 随机选择几个预训练模型：

```python
model_names = [
    'nin_cifar10',
    'resnet20_cifar10',
    'preresnet20_cifar10'
]
```
- 补全`class ensembleNet`。


```python
from pytorchcv.model_provider import get_model as ptcv_get_model

class ensembleNet(nn.Module):
  def __init__(self, model_names):
    super().__init__()
    self.models = nn.ModuleList([ptcv_get_model(name, pretrained=True) for name in model_names])
    self.softmax = nn.Softmax(dim=1)
  def forward(self, x):
    ensemble_logits = None
    for i, m in enumerate(self.models):
      # TODO: sum up logits from multiple models
      # return ensemble_logits
      logits = m(x)
      if ensemble_logits is None:
        ensemble_logits = logits
      else:
        ensemble_logits += logits

      ensemble_logits /= len(self.models)

    return self.softmax(ensemble_logits)
```
- 实现'Ensemble attack with IFGSM'；


```python
adv_examples, ifgsm_acc, ifgsm_loss = gen_adv_examples(ensemble_model, adv_loader, ifgsm, loss_fn)
print(f'ifgsm_ensemble_acc = {ifgsm_acc:.5f}, ifgsm_ensemble_loss = {ifgsm_loss:.5f}')

create_dir(root, 'ifgsm_ensemble', adv_examples, adv_names)
```

### Strong baseline (acc <= 0.30)
助教提供了两个思路：
1. Ensemble Attack + paper B (pick right models) + IFGSM
2. Ensemble Attack + many models + MIFGSM

Ensemble Attack与Medium Baseline相同。

思路1'paper B (pick right models)'使用了论文
[Query-Free Adversarial Transfer via Undertrained Surrogates](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2007.00806)的思想，旨在挑选更effective的单个代理模型，而不是随机的一组代理模型。论文定义了**Undertrained Models**，
它包含两个condition：
- 具有更高验证集损失；
- 训练的step或epoch更少；

可产生更强的可转移的对抗性攻击。

**实作中采用了思路2的方法。**

mifgsm_acc = 0.00000, mifgsm_loss = 2.36389

- 增加更多代理模型；


```python
model_names = [
    'nin_cifar10',
    'resnet20_cifar10',
    'resnet56_cifar10',
    'preresnet20_cifar10',
    'preresnet56_cifar10',
    'seresnet20_cifar10',
    'seresnet56_cifar10',
    'sepreresnet20_cifar10',
    'sepreresnet56_cifar10',
    'wrn16_10_cifar10',
    'wrn20_10_1bit_cifar10',
    'rir_cifar10',
    'diaresnet20_cifar10',
    'diapreresnet20_cifar10',
    'densenet40_k12_cifar10',
]
```

- 实现MIFGSM算法

MIFGSM（Momentum Iterative Fast Gradient Sign Method）对抗攻击算法是IFGSM引入动量的优化版本，使得攻击更稳定、迁移性更强。

```python
def mifgsm(model, x, y, loss_fn, epsilon=epsilon, alpha=alpha, num_iter=50, mu=1.0):
    x_adv = x
    # initialze momentum tensor
    momentum = torch.zeros_like(x).detach().to(device)
    # write a loop of num_iter to represent the iterative times
    for i in range(num_iter):
        x_adv = x_adv.detach().clone()
        x_adv.requires_grad = True # need to obtain gradient of x_adv, thus set required grad
        loss = loss_fn(model(x_adv), y) # calculate loss
        loss.backward() # calculate gradient

        # TODO: Momentum calculation
        grad = x_adv.grad.detach()
        # 梯度归一化（L1范数）
        grad_norm = torch.mean(torch.abs(grad), dim=(1, 2, 3), keepdim=True)
        normalized_grad = grad / (grad_norm + 1e-8)  # 避免除零
        grad = mu * momentum + normalized_grad
        momentum = grad

        x_adv = x_adv + alpha * grad.sign()
        x_adv = torch.max(torch.min(x_adv, x+epsilon), x-epsilon) # clip new x_adv back to [x-epsilon, x+epsilon]
    return x_adv
```

### Boss baseline (acc <= 0.15)
根据助教提示，实现DIM-MIFGSM算法。

dim_mifgsm_acc = 0.00000, dim_mifgsm_loss = 2.34545

DIM（Diverse Input Method）在攻击过程中对输入图像进行随机resize + padding的处理，提高攻击的迁移性。

我们实现DIM机制，并在Strong Baseline基础上修改MIFGSM算法。


```python
# DIM + MI-FGSM

import random
import torchvision.transforms.functional as TF

def input_diversity(x, resize_rate=0.9, diversity_prob=0.5):
  """
  对输入图像进行随机resize + padding（DIM核心步骤）
  """
  if random.random() < diversity_prob:
    img_size = x.shape[-1]
    new_size = int(img_size * resize_rate)
    rescaled = TF.resize(x, [new_size, new_size])

    pad_top = random.randint(0, img_size - new_size)
    pad_bottom = img_size - new_size - pad_top
    pad_left = random.randint(0, img_size - new_size)
    pad_right = img_size - new_size - pad_left

    padded = TF.pad(rescaled, 
     [pad_left, pad_top, pad_right, pad_bottom],
      fill=0)
    return padded
  else:
    return x

def dim_mifgsm(model, x, y, loss_fn, epsilon=epsilon, alpha=alpha, num_iter=50, mu=1.0, diversity_prob=0.7):
    x_adv = x
    # initialze momentum tensor
    momentum = torch.zeros_like(x).detach().to(device)
    # write a loop of num_iter to represent the iterative times
    for i in range(num_iter):
        x_adv = x_adv.detach().clone()
        x_adv.requires_grad = True # need to obtain gradient of x_adv, thus set required grad

        diversified_x = input_diversity(x_adv, diversity_prob=diversity_prob)

        loss = loss_fn(model(diversified_x), y) # calculate loss
        loss.backward() # calculate gradient

        # TODO: Momentum calculation
        grad = x_adv.grad.detach()
        # 梯度归一化（L1范数）
        grad_norm = torch.mean(torch.abs(grad), dim=(1, 2, 3), keepdim=True)
        normalized_grad = grad / (grad_norm + 1e-8)  # 避免除零
        grad = mu * momentum + normalized_grad
        momentum = grad

        x_adv = x_adv + alpha * grad.sign()
        x_adv = torch.max(torch.min(x_adv, x+epsilon), x-epsilon) # clip new x_adv back to [x-epsilon, x+epsilon]
    return x_adv

```

## Code

[代码实现](https://github.com/Aaricis/Hung-yi-Lee-ML2022/tree/main/HW10)

## Report

### Part 1: Attack

**根據你最好的實驗結果，簡述你是如何產生transferable noises, Judge Boi上Accuracy降到多少?**

> 结果需提交到Judge Boi才能看到Accuracy，非台大的学生不能提交。

理论上讲，最好的实验结果应该是DIM-MIFGSM算法。代理模型选择借鉴了论文[Query-Free Adversarial Transfer via Undertrained Surrogates](https://www.google.com/url?q=https%3A%2F%2Flink.zhihu.com%2F%3Ftarget%3Dhttps%3A%2F%2Farxiv.org%2Fabs%2F2007.00806)的实验，主要采用ResNet，SENet，DenseNet类型的预训练模型，也包括了其他类型的模型，详情如下：

```python
model_names = [
    'nin_cifar10',
    'resnet20_cifar10',
    'resnet56_cifar10',
    'preresnet20_cifar10',
    'preresnet56_cifar10',
    'seresnet20_cifar10',
    'seresnet56_cifar10',
    'sepreresnet20_cifar10',
    'sepreresnet56_cifar10',
    'wrn16_10_cifar10',
    'wrn20_10_1bit_cifar10',
    'rir_cifar10',
    'diaresnet20_cifar10',
    'diapreresnet20_cifar10',
    'densenet40_k12_cifar10',
]
```

### Part 2: Defense 

**當source model為resnet110_cifar10(from Pytorchcv), 使用最原始的fgsm 攻擊在dog2.png的圖片。** 

1. **請問被攻擊後的預測的class是錯誤的嗎？(1pt) 有的話：變成哪個class? 沒有的話：則不用作答** 

   是错误的，变成cat。

2. **實作jpeg compression (compression rate=70%) 前處理圖片, 請問 prediction class是錯誤的嗎？同第一題作答 (1pt)** 

   prediction class是正确的，依然为dog。

3. **Jpeg compression為什麼可以抵擋adversarial attack, 讓模型維持高正確率？ (1pt)** 

   - [ ] 圖片壓縮時讓色彩更鮮豔
   - [x] 圖片壓縮時把雜訊減少 
   - [ ] 圖片壓縮讓圖片品質下降 
   - [ ] 圖片壓縮時雜訊反而變大

## Reference

[Attack（李宏毅）机器学习 2023 Spring HW10 - 知乎](https://zhuanlan.zhihu.com/p/14931222760)

[李宏毅2022机器学习HW10解析 - 知乎](https://zhuanlan.zhihu.com/p/537468028)
