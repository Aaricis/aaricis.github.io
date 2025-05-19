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



## Code



## Report



## Reference

[DI-2-FGSM](https://github.com/cihangxie/DI-2-FGSM)
