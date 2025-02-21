---
title: Phoneme Classification
date: 2025-02-16 15:07:00 +/-8
categories: [Machine Learning 2022]
tags: [Hung-yi Lee]     # TAG names should always be lowercase
---

## Objectives

- Data Preprocessing：从waveform中抽取MFCC特征；
- Classification：使用预提取的MFCC特征进行phoneme分类；

## Task Introduction

**Multiclass Classification**：预测speech中每个phoneme所属的类别。

## 思路&Code

[双过Boss Baseline](https://github.com/Aaricis/Hung-yi-Lee-ML2022/tree/main/HW2)

## Report Questions

**1. (2%) Implement 2 models with approximately the same number of parameters, (A) one narrower and deeper (e.g. hidden_layers=6, hidden_dim=1024) and (B) the other wider and shallower (e.g. hidden_layers=2, hidden_dim=1700). Report training/validation accuracies for both models.**

> 1. 实现两个参数量大致相同的模型，(A) 一个深窄的（例如，隐藏层数=6，隐藏维度=1024），(B) 一个浅宽的（例如，隐藏层数=2，隐藏维度=1750）。报告两个模型的训练/验证准确率。

计算神经网络的参数量：

以全连接层为例：

假设 输入神经元数为M，输出神经元数为N，则

（1）bias为True时：

则参数数量为：M*N + N（bias的数量与输出神经元数的数量是一样的）

（2）bias为False时：

则参数数量为：M×N

使用Pytorch直接计算模型的参数量：

```python
model = Classifier(input_dim=input_dim, hidden_layers=hidden_layers, hidden_dim=hidden_dim)
total_params = sum(param.numel() for param in model.parameters())
print(f'Total params: {total_params}')
```

| hidden_layers | hidden_dim | Total params | training/validation accuracies |
| ------------- | ---------- | ------------ | ------------------------------ |
| 6             | 1024       | 6380585      | 0.478697/0.470                 |
| 2             | 1760       | 6341321      | 0.486987/0.471                 |

PS: 其他参数跟Sample Code保持一致

**2. (2%) Add dropout layers, and report training/validation accuracies with dropout rates equal to (A) 0.25/(B) 0.5/(C) 0.75 respectively.**

> 1. 添加dropout层，并分别报告dropout率为(A) 0.25/(B) 0.5/(C) 0.75时的训练/验证准确率。

参数跟Sample Code保持一致，模型增加Dropout层

| dropout率 | training/validation accuracies |
| --------- | ------------------------------ |
| 0.25      | 0.444572/0.453                 |
| 0.5       | 0.427810/0.443                 |
| 0.75      | 0.397413/0.421                 |



## 参考资料

[课程官方资料库（含PPT和样例代码）](https://speech.ee.ntu.edu.tw/~hylee/ml/2022-spring.php)

[李宏毅机器学习 hw2 boss baseline 解析_李宏毅2021 hw02multiclass classification-CSDN博客](https://blog.csdn.net/qq_43613342/article/details/127007955)

[【李宏毅机器学习HW2】_李宏毅2022hw2-CSDN博客](https://blog.csdn.net/detemination_/article/details/127194301)

[李宏毅机器学习hw2 Boss baseline（2023）_李宏毅机器学习2023作业hw2-CSDN博客](https://blog.csdn.net/asf2013/article/details/136402896)