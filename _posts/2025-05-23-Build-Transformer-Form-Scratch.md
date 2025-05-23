---
title: Build Transformer Form Scratch
date: 2025-05-23 17:30:00 +/-8
categories: [LLM]
tags: [Transformer]     # TAG names should always be lowercase
math: true
---

为了弄清楚Transformer的实现细节，这里从零开始复现论文[Attention Is All You Need](https://arxiv.org/abs/1706.03762)中执行文本翻译任务的Transformer(Pytorch)。该版本与Pytorch提供的`torch.nn.Transformer`不同略有不同，包含了Embedding的步骤。