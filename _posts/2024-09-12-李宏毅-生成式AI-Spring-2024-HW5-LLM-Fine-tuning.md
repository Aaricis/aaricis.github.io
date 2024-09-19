---
title: 【李宏毅-生成式AI】Spring 2024, HW5：LLM Fine-tuning
date: 2024-09-12 16:40:00 +/-8
categories: [LLM, Spring 24 GenAI]
tags: [Hung-yi Lee]     # TAG names should always be lowercase
---

#  Task Overview

训练一个会写唐诗的AI模型。给定AI模型诗的前两句，写出诗的后两句。

原本的LLM不具备写诗的能力。我们用ChatGPT和kimi都试一下👇，它们无一例外都输出了对诗的鉴赏。

![](../assets/images/Hung-yi_Lee/hw5-1.png)

![](../assets/images/Hung-yi_Lee/hw5-2.png)

没有经过Fine-tuning的模型，不具备写唐诗的能力。我们的目的是教AI模型写唐诗。

![](../assets/images/Hung-yi_Lee/hw5-3.png)





#  Model and Dataset

## Model

实验提供了两个70亿参数的模型可供选择：

1. Taide-7B：Taide7B模型是“可信AI对话引擎”（TAIDE）项目的一部分，主要为台湾开发。该模型基于LLaMa模型，专注于处理繁体中文任务，包括翻译、摘要、信件写作和文章生成。
2. MediaTek Breeze 7B：MR Breeze-7B 是联发科旗下研究机构联发科技研究中心（MediaTek Research）开发的一款全新开源大语言模型（LLM），专为处理繁体中文和英文而设计。这款模型拥有70亿个参数，基于广受赞誉的Mistral模型进行设计和优化。

## Dataset

专门用于微调LLM的唐诗数据集 [Tang poem dataset](https://github.com/CheeEn-Yu/GenAI-Hw5)，里面包含5000首诗。

![](../assets/images/Hung-yi_Lee/hw5-4.png)

dataset主要包含两个JSON文件：

- Tang_testing_data.json：测试集，包含15条数据
- Tang_training_data.json：训练集，包含5001条数据

训练集数据如上图所示，包含`instruction`, `input`, `output`；测试集只包含`instruction`, `input`，答案在Tang_tesing_gt.txt文件中。

# Changing the Generation Behavior:Decoding Parameters

生成式模型选择下一个token的方法是：从下一个token的分布中采样。

![](../assets/images/Hung-yi_Lee/hw5-5.png)

通过改变采样方式，可以改变语言模型生成下一个token的方式。

我们可以调整模型超参数，控制模型的行为。让模型的输出：longer vs. shorter; diverse vs.static；超参数有：

- temperature
- Top-k
- Top-p
- max_length

## Temperature

temperature控制模型输出的diversity。它改变了数据的分布概率，temperature越小，模型的输出越固定；temperature越大，模型的输出越随机，输入同样prompt，模型的输出差异很大。

![](../assets/images/Hung-yi_Lee/hw5-6.png)