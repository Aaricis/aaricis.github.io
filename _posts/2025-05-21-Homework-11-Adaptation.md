---
title: 李宏毅-ML2022-HW11-Adaptation
date: 2025-05-21 15:07:00 +/-8
categories: [Machine Learning 2022]
tags: [Hung-yi Lee]     # TAG names should always be lowercase
math: true
---

## Task Description

本次作业的主题是**Domain Adaptation**，即**领域自适应**。

假设你要执行与真实3D场景相关的任务，但是真实环境3D图像很难标记并且价格昂贵，而模拟图像（例如GTA-5上的模拟场景）易于标记。如果将模拟图像作为训练集，真实环境图像作为测试集，这样作会有什么问题？

模型会将真实环境图像识别为“异常”，因为训练数据和测试数据来自不同的domain。

如何解决这个问题哪？需要进行**Domain Shift**，我们旨在找到一个特征提取器，它接收输入数据并输出特征空间。这个特征提取器能够滤掉domain相关的信息，只保留不同domain之间共享的特征（[详情参见课程录影](https://youtu.be/Mnk_oUrgppM)）。

**具体的任务是**：给定真实图像（with labels）和涂鸦图像（without labels），使用Domain Adaptation技术预测手绘图像的label。

## Dataset

- Label: 10 classes (numbered from 0 to 9). 
- Training : 5000 (32, 32) RGB real images (with label). 
- Testing : 100000 (28, 28) gray scale drawing images.

## Data Format

Unzip real_or_drawing.zip, the data format is as below: 

- real_or_drawing/ 
  - train_data/ 
    - 0/ 
      - 0.bmp, 1.bmp … 499.bmp 
    - 1/ 
      - 500.bmp, 501.bmp … 999.bmp 
    - … 9/ 
  - test_data/
    - 0/ 
      - 00000.bmp
      - 00001.bmp 
      - … 99999.bmp

## 思路





## Code



## Report



## Reference

