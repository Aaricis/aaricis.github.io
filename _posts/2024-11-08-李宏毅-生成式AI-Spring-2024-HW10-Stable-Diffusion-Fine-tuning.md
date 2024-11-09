---
title: 【李宏毅-生成式AI】Spring 2024, HW10：Stable Diffusion Fine-tuning
date: 2024-11-08 16:40:00 +/-8
categories: [LLM, Spring 24 GenAI]
tags: [Hung-yi Lee]     # TAG names should always be lowercase
---

## Overview

### Text-to-Image Model

Text-to-Image Model可以生成与文本描述相匹配的图像。ChatGPT-4O具备Text-to-Image能力，我们用ChatGPT-4O试一下，输入文字描述，模型根据文字描述输出图片👇

![](../assets/images/Hung-yi_Lee/hw10-1.png)

![](../assets/images/Hung-yi_Lee/hw10-2.png)

### Personalization

公共基础模型不能满足个性化的要求，例如ChatGPT-4O每次输出的人脸都不一样。想要模型每次输出特定的人物形象，例如每次都生成Brad Pitt的脸，需要对模型进行微调。

![](../assets/images/Hung-yi_Lee/hw10-3.png)

### Achieve Personalization Using LoRA

使用LoRA（Low-Rank Adaptation）对基础模型进行微调。LoRA是微软在2021年推出的微调方法，它冻结预训练的模型权重，将可训练的秩分解矩阵注入到Transformer架构的每一层，从而大大减少下游任务的可训练参数量。多数情况下，LoRA的性能可与完全微调的模型相媲美，并且不会引入额外的推理时延。

神经网络进行矩阵乘法时，它的权重矩阵是full-rank的。在适配特定任务时，预训练的语言模型具有较低的”内在维度（intrinsic dimension）“，在随机投射的更小子空间上任然可以有效的学习。”内在维度（intrinsic dimension）“指矩阵中存在有效信息的维度，小于等于矩阵的实际维度。受此启发，LoRA假设在设配特定任务时，矩阵权重更新具有较低的内在维度，即对模型微调就是调整这些低秩的内在维度。

![](../assets/images/Hung-yi_Lee/hw10-4.png)

在训练期间，LoRA在原本的矩阵$W\in R^{d\times k}$旁边插入一个并行的权重矩阵$\Delta W \in R^{d \times k}$。因为模型的低秩性，$\Delta W$可被拆分为矩阵$B\in R^{d \times r}$和$A\in R^{r\times k}$，其中$r\ll min(d, k)$。
