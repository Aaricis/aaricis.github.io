---
title: Self-attention
date: 2025-03-09 15:07:00 +/-8
categories: [Machine Learning 2022]
tags: [Hung-yi Lee]     # TAG names should always be lowercase
---

## Objective

- 了解Self-attention；

- 学会使用Transformer。

## Task Description

Speaker Identification（语者识别）：多分类任务，从给定的语音中预测演讲者的类别。

## Dataset 

VoxCeleb2

- Training: 56666 processed audio features with labels. 
- Testing: 4000 processed audio features (public & private) without labels. 
- Label: 600 classes in total, each class represents a speaker.

## 思路

### Simple Baseline(> 0.60824)
直接跑一遍Simple Code（Score: 0.60500
Private score: 0.61300）不能达到。

Score: 0.65375
Private score: 0.65625

模型使用两层Transformer👇
```python
self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=2)
```
forward中使用encoder
```python
# out = self.encoder_layer(out)
out = self.encoder(out)
```

### Medium Baseline(> 0.70375)
Score: 0.76825
Private score: 0.76925

- 增加`d_model`的值，因为输出维度n_spks=600差距过大；
```python
d_model=224
```
- `nn.TransformerEncoderLayer`增加`dropout`参数，`dropout=0.2`；
```python
self.encoder_layer = nn.TransformerEncoderLayer(
			d_model=d_model, dim_feedforward=256, nhead=2, dropout=dropout
		)
```
- 堆叠3个transformer层；
```python
self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=3)
```
- 输出层改为一层并增加batchnorm机制；
```python
self.pred_layer = nn.Sequential(
      nn.BatchNorm1d(d_model),
      nn.Linear(d_model, n_spks),
    )
```

### Strong Baseline(>0.77750)
Score: 0.85025
Private score: 0.85025

使用 `torchaudio.models.Conformer` 作为编码器代替Transformer，提取序列特征。
ref:https://pytorch.org/audio/stable/generated/torchaudio.models.Conformer.html

注意:
- Conformer的输入形状为(batch_size, seq_len, d_model)，输出形状相同；
- `Conformer.forward()`需要额外输入`lengths`,是一个张量，表示每个输入序列的实际长度，以便模型在处理时可以忽略填充部分（padding）；
```python
lengths = torch.full((out.shape[0],),out.shape[1])
```
  (1)out.shape: out 是经过 prenet 处理后的张量，形状为 (batch_size, seq_len, d_model)。

  - batch_size：批次大小。

  - seq_len：序列长度（时间步数）。

  - d_model：特征维度。

  - out.shape[0] 是批次大小（batch_size）。

  - out.shape[1] 是序列长度（seq_len）。

  (2)torch.full: torch.full 是 PyTorch 中的一个函数，用于创建一个填充了指定值的张量。

  语法：torch.full(size, fill_value, ...)。

  size：张量的形状。

  fill_value：填充的值。

  (3) (out.shape[0],):
  - 这是一个元组，表示张量的形状。
  - (out.shape[0],) 表示一个一维张量，长度为 batch_size。

  (4) out.shape[1]:
  这是填充的值，表示序列的长度（seq_len）。

```python
import torch
import torch.nn as nn
import torchaudio.models

class Classifier(nn.Module):
    def __init__(self, d_model=224, n_spks=600, dropout=0.2):
        super().__init__()
        # 将输入维度从 40 映射到 d_model
        self.prenet = nn.Linear(40, d_model)

        # 使用 Conformer 代替 Transformer
        self.encoder = torchaudio.models.Conformer(
            input_dim=d_model,  # 输入特征维度
            num_heads=2,  # 注意力头数
            ffn_dim=256,  # Feed-forward 层隐藏维度
            num_layers=3,  # Conformer 层数
            depthwise_conv_kernel_size=31,  # 深度卷积核大小
            dropout=dropout,
        )

        # 预测层（BatchNorm + 线性分类层）
        self.pred_layer = nn.Sequential(
            nn.BatchNorm1d(d_model),
            nn.Linear(d_model, n_spks),
        )

    def forward(self, mels):
        """
        args:
            mels: (batch_size, seq_len, 40)
        return:
            out: (batch_size, n_spks)
        """
        # 线性变换调整维度
        out = self.prenet(mels)  # (batch_size, seq_len, d_model)

        # Conformer 期望输入格式 (batch_size, seq_len, d_model)，无需 permute
        lengths = torch.full((out.shape[0],),out.shape[1]).to(device)
        out, _ = self.encoder(out, lengths)

        # Mean Pooling 取全局信息
        stats = out.mean(dim=1)  # (batch_size, d_model)

        # 通过分类层
        out = self.pred_layer(stats)  # (batch_size, n_spks)

        return out

```

### Boss Baseline(>0.86500)
Score: 0.87575
Private score: 0.87250

- 根据助教提示，直接加上Self-Attention Pooling和Additive Margin Softmax,训练15万个step，结果就直接惨掉（Score: 0.83900 Private score: 0.83900），比Strong Baseline的score还不如。

- 只使用Conformer+Self-Attention Pooling，结果略有提升，但是还不能达到Boss Baseline(Score: 0.85550 Private score: 0.85550)。Additive Margin Softmax看起来实际没有起到什么作用。

- 调整Conformer头数、层数, dropout等参数，score略有波动，但不明显。

- 模型增加BatchNorm层、l2正则，改进Pooling层（Mean + Self-Attention）让池化更稳定，score略有增加。

至此，单模还是不能达到Boss Baseline，跟助教的说法不一样，不知道是代码还是参数的问题，如果能有改进的建议，非常感谢 : )

最终只好使用投票法Ensemble，才超过Boss Baseline。

- **Self-Attention Pooling**

```python

class SelfAttentionPooling(nn.Module):
    """
    Mean Pooling + Self-Attention Pooling
    """
    def __init__(self, input_dim):
        super(SelfAttentionPooling, self).__init__()
        self.attention = nn.Linear(input_dim, 1)

    def forward(self, batch_rep):
        """
        input:
            batch_rep : size (N, T, H), N: batch size, T: sequence length, H: Hidden dimension

        return:
            utter_rep: size (N, H)
        """
        # Self-Attention Weights
        att_w = F.softmax(self.attention(batch_rep), dim=1)  # (N, T, 1)
        att_out = torch.sum(batch_rep * att_w, dim=1)  # Weighted sum (N, H)
        
        # Combine both
        utter_rep = att_out
        return utter_rep

```

- **Additive Margin Softmax**

```python

class AMSoftmax(nn.Module):
    def __init__(self, in_features, n_classes, s=20, m=0.2): # s=30, m=0.4
        super(AMSoftmax, self).__init__()
        self.in_features = in_features
        self.n_classes = n_classes
        self.s = s  # 缩放因子
        self.m = m  # 角度边距
        self.weight = nn.Parameter(torch.FloatTensor(n_classes, in_features))
        nn.init.xavier_uniform_(self.weight)  # 初始化权重

    def forward(self, x, labels=None):
        """
        x: (batch_size, in_features) 输入特征
        labels: (batch_size,) 目标标签 (训练模式) 或 None (推理模式)
        return: (batch_size, n_classes) AM-Softmax logits
        """
        # 归一化输入和权重
        x = F.normalize(x, p=2, dim=-1)  # 归一化输入特征
        weight = F.normalize(self.weight, p=2, dim=-1)  # 归一化权重

        # 计算余弦相似度
        cosine = F.linear(x, weight)  # (batch_size, n_classes)

        if labels is not None:
            # 训练模式：计算 AM-Softmax
            one_hot = F.one_hot(labels.to(torch.int64), num_classes=self.n_classes).float()
            cosine_m = cosine - self.m * one_hot  # 仅对目标类施加 margin
            logits = self.s * cosine_m
        else:
            # 推理模式：仅进行归一化计算
            logits = self.s * cosine  # 直接返回 softmax logits

        return logits

```

## Code

[双过Boss Baseline](https://github.com/Aaricis/Hung-yi-Lee-ML2022/blob/main/HW4/hw04-speaker-identification.ipynb)

## Report

**1. Make a brief introduction about a variant of Transformer. (2 pts)** 

Conformer是一种结合了Transformer和卷积神经网络（CNN）的模型，主要用于处理序列数据（如语音、文本等）。它通过巧妙地融合Transformer地自注意力机制和CNN地局部特征提取能力，在语音识别任务中取得了显著效果。

**2. Briefly explain why adding convolutional layers to Transformer can boost performance. (2 pts)**

全局与局部特征的结合：

- Transformer 的自注意力机制擅长捕捉全局依赖关系。
- CNN 的卷积模块擅长捕捉局部特征。
- Conformer 通过结合两者，能够同时建模序列中的全局和局部信息。

## Reference

[李宏毅深度学习 2021 作业四 Self-Attention 实验记录 - Niku's Blog](https://www.nikunokoya.com/posts/lhy2021_hw4/#conformer)

