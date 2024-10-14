---
title: 【李宏毅-生成式AI】Spring 2024, HW6：Learning from Human Preference
date: 2024-10-10 16:40:00 +/-8
categories: [LLM, Spring 24 GenAI]
tags: [Hung-yi Lee]     # TAG names should always be lowercase
---

# Task Overview

使用RLHF(Reinforcement Learning from Human Feedback)训练LLM，使它输出符合人类偏好的回答。这个过程也称为‘Alignment’，中文翻译作‘对齐’。具体‘对齐’什么东西呢？‘对齐’人类的喜好和偏向。

本次作业的目标是训练一个”支持漫画真人化“的LLM。如下图所示，在Alignment之前，模型输出”中立“的回答，不具有明显的倾向。我们希望Alignment之后，模型支持漫画真人化。

![](../assets/images/Hung-yi_Lee/hw6-1.PNG)

# Reinforcement Learning from Human Feedback (RLHF)

## 什么是RLHF？

RLHF是LLM训练的第三阶段，目的是使LLM输出满足人类偏好的回答。

![](../assets/images/Hung-yi_Lee/hw6-3.PNG)

我们向ChatGPT提问，让它生成多个回答，GPT会让人类反馈”哪个答案更好？“。GPT收到反馈信息之后，微调模型参数，提高输出”好“回答的概率。

![](../assets/images/Hung-yi_Lee/hw6-2.PNG)

RLHF涉及到多模型训练过程和不同的部署阶段，训练过程可被分解为三个核心步骤：

![Media: Methods > Media Item > Light mode](https://images.ctfassets.net/kftzwdyauwt9/12CHOYcRkqSuwzxRp46fZD/928a06fd1dae351a8edcf6c82fbda72e/Methods_Diagram_light_mode.jpg?w=3840&q=90&fm=webp)

1. 使用监督学习微调语言模型(LM)；
2. 收集比较数据，训练奖励模型(Reward model)；
3. 使用强化学习算法对监督策略进行微调以优化该奖励；

## Step1：预训练语言模型

本作业直接使用联发科的Breeze-7b模型作为预训练的LM。

**Breeze-7B** 是联发科（MediaTek）旗下研究机构开发的一款**大语言模型（LLM）**，专为处理英文和繁体中文的任务而优化。它拥有 **70亿参数**，并在繁体中文的多个基准测试中表现出色，尤其在与 **GPT-3.5** 等强大模型的对比中展现了较高的竞争力。

## Step2：训练奖励模型

***为什么需要奖励模型？***

*评价LM输出答案是否满足人类的偏好，需要人类提供回馈。由于人类的时间精力优先，我们使用RM来模仿人类的行为。*

- 收集比较数据；
- 给定一个问题，会有多个回答，人类会对这些回答进行排序；
- 奖励模型学习哪种反应更好（更类似于人类偏好）；

本次作业使用DPO(Direct Preference Optimization)对齐LLM。

RLHF需要额外训练奖励模型，并且强化学习训练非常不稳定，超参数难以调整。DPO通过loss function直接从偏好数据优化LLM。

![](../assets/images/Hung-yi_Lee/hw6-4.PNG)



# 参考

[Instruct GPT](https://arxiv.org/abs/2203.02155)
