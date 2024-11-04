---
title: 【李宏毅-生成式AI】Spring 2024, HW9：Quick Summary of Lecture Video
date: 2024-11-03 16:40:00 +/-8
categories: [LLM, Spring 24 GenAI]
tags: [Hung-yi Lee]     # TAG names should always be lowercase
---

## Task Introduction

本次作业的任务是：快速总结讲座视频。给定一个讲座视频，首先使用自动语音识别(**automatic speech recognition** (ASR))将视频转化为逐字稿；然后使用LLM对逐字稿做摘要。

![](../assets/images/Hung-yi_Lee/hw9-1.png)

作业使用的视频是[Lin-shan Lee](https://linshanlee.com/)教授2023年的讲座”[Signals and Life（信号与人生）](https://www.youtube.com/watch?v=MxoQV4M0jY8)“。由于原始视频很长，作业使用1:43:24到2:00:49的片段。

## Task Pipeline

本次作业的Pipeline分为两个阶段：

1. **Automatic Speech Recognition**（自动语音识别）：使用OpenAI的Whisper模型进行语音识别，将视频转化成逐字稿。
2. **Summarization**（摘要）：为LLM设计一个prompt，将逐字稿总结为300-500的繁体中午摘要。

### Automatic Speech Recognition

语音识别是将语音信号转化为书面文本的过程。

![](../assets/images/Hung-yi_Lee/hw9-2.PNG)

### Whisper — Introduction

OpenAI Whisper是一个能够准确的将口语转录并翻译成文本的模型。Whisper经过了68万小时的多语言多任务监督数据的训练，这些数据来自互联网，包括99种不同的语言。多任务训练数据包含4种任务：

- English transcription：英语转录；
- Any-to-English speech translation：任何语言的语音转录为英文文本；
- Non-English transcription：非英文转录为英文文本；
- No speech：背景音不会转录为任何文字。



![](../assets/images/Hung-yi_Lee/hw9-3.PNG)

OpenAI Whisper既可以作转录也可以作翻译，本次作业使用Whisper的转录功能，将中文语音转化为中文文本。

![](../assets/images/Hung-yi_Lee/hw9-4.PNG)

### Summarization

为LLM设计一个恰当的prompt，将文本总结为300-500字的繁体中文摘要。

#### Methods

最直接的方法是直接将未处理的文本输入LLM做摘要。但是，如果文本太长，LLM无法一次性处理全部内容。因此，本次作业提供两种做摘要的方法：

- Multi-Stage Summarization；
- Refinement；

##### Multi-Stage Summarization

将长文本分割为多个小片段，分别获取每个片段的摘要，然后处理这些摘要生成最终的摘要。

![](../assets/images/Hung-yi_Lee/hw9-5.PNG)

##### Refinement

- 将长文本分为多个段落；
- 首先，总结第一段；
- 将第一段的摘要与第二段一起作摘要；
- 将前两段的摘要和第三段一起作摘要；
- 持续这个过程，直到整个长文本被总结完。

![](../assets/images/Hung-yi_Lee/hw9-6.PNG)