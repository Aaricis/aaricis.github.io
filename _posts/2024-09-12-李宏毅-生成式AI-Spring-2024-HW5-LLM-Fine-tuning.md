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

没有经过Fine-tuning的模型，不具备写唐诗的能力。

#  Model and Dataset

## Model

实验提供了两个70亿参数的模型可供选择：

1. Taide-7B：Taide7B模型是“可信AI对话引擎”（TAIDE）项目的一部分，主要为台湾开发。该模型基于LLaMa模型，专注于处理繁体中文任务，包括翻译、摘要、信件写作和文章生成。
2. MediaTek Breeze 7B