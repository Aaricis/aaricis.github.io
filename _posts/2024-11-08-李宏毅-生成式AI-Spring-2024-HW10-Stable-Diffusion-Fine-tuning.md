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

基础模型不能满足个性化的要求，例如ChatGPT-4O每次输出的人脸都不一样。想要模型每次输出特定的人物形象，例如每次都生成Brad Pitt的脸，需要对模型进行微调。

![](../assets/images/Hung-yi_Lee/hw10-3.png)

## Achieve Personalization Using LoRA

###  LoRA

使用LoRA（Low-Rank Adaptation）对基础模型进行微调。LoRA是微软在2021年推出的微调方法，它冻结预训练的模型权重，将可训练的秩分解矩阵注入到Transformer架构的每一层，从而大大减少下游任务的可训练参数量。多数情况下，LoRA的性能可与完全微调的模型相媲美，并且不会引入额外的推理时延。

神经网络进行矩阵乘法时，它的权重矩阵是full-rank的。在适配特定任务时，预训练的语言模型具有较低的”内在维度（intrinsic dimension）“，在随机投射的更小子空间上任然可以有效的学习。”内在维度（intrinsic dimension）“指矩阵中存在有效信息的维度，小于等于矩阵的实际维度。受此启发，LoRA假设在设配特定任务时，矩阵权重更新具有较低的内在维度，即对模型微调就是调整这些低秩的内在维度。

![](../assets/images/Hung-yi_Lee/hw10-4.png)

LoRA在原本的矩阵$W\in R^{d\times k}$旁边插入一个并行的权重矩阵$\Delta W \in R^{d \times k}$。因为模型的低秩性，$\Delta W$可被拆分为矩阵$B\in R^{d \times r}$和$A\in R^{r\times k}$，其中$r\ll min(d, k)$，从而实现了极小的参数数量训练LLM。在训练期间，$W$被冻结，不会接受梯度更新，而$A$和$B$包含可训练的参数会被更新。$W$和$\Delta W$接受相同的输入$x$，训练完成后各自的输出向量按位置相加，因此不会产生额外的推理时间，如下式所示：
$$
h = W_0 x + \Delta W x = W_0 x + BAx
$$
在训练开始的时候，对矩阵$A$进行随机高斯初始化，矩阵$B$使用零矩阵初始化。因为$r$通常是一个非常小的值（实验证明1、2、4、8的效果就非常好），所以LoRA在训练时引入的参数量是非常小的，因此它的训练非常高效，不会带来显著的显存增加。

### Fine-tuning with LoRA

使用LoRA微调基础模型，获得额外的personalization能力。

![](../assets/images/Hung-yi_Lee/hw10-5.PNG)

## Goal of This Homework

使用同一个人的面部图片微调Stable Diffusion模型，使它生成面孔一致的图片。

![](../assets/images/Hung-yi_Lee/hw10-6.PNG)

原始的Stable Diffusion模型每次产生的图片人脸都不一样。我们使用Brad Pitt的图片和对应的文本描述微调原始模型，使它产生的图片都是Brad Pitt的脸。

给定充分的训练时间，Stable Diffusion生成的图片将会拥有同一个人的脸。

![](../assets/images/Hung-yi_Lee/hw10-7.PNG)

## Evaluation

将给定的prompt输入微调后的模型生成图片，最终的得分将会从三个方面进行评估：

- 图片是否与训练数据相似？
- 图片和文字是否匹配？
- 图片是否包含人脸？

## Task Introduction

![](../assets/images/Hung-yi_Lee/hw10-8.PNG)

## Step 1. Fine-tune Stable Diffusion

使用Hugging Face`stablediffusionapi/cyberrealistic-41`模型作为base model；训练数据为100张Brad Pitt的照片和对应的文本描述。如下图所示，照片和文本描述成对出现，具有相同的文件名。

![](../assets/images/Hung-yi_Lee/hw10-9.png)



**安装必要的库：**

```python
# Install the required packages
os.chdir(root_dir)
!pip -q install timm==1.0.7
!pip -q install fairscale==0.4.13
!pip -q install transformers==4.41.2
!pip -q install requests==2.32.3
!pip -q install accelerate==0.31.0
!pip -q install diffusers==0.29.1
!pip -q install einop==0.0.1
!pip -q install safetensors==0.4.3
!pip -q install voluptuous==0.15.1
!pip -q install jax==0.4.33
!pip -q install peft==0.11.1
!pip -q install deepface==0.0.92
!pip -q install tensorflow==2.17.0
!pip -q install keras==3.2.0
```

**导入必要的包：**

```python
#@markdown ##  Import necessary packages
#@markdown It is recommmended NOT to change codes in this cell.
import argparse
import logging
import math
import os
import random
import glob
import shutil
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers

# Python Imaging Library（PIL）图像处理
from PIL import Image

# 图像处理
from torchvision import transforms
from torchvision.utils import save_image

# 显示进度条
from tqdm.auto import tqdm

# Parameter-Efficient Fine-tuning(PEFT)库
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict

# Hugging Face transformers
from transformers import AutoProcessor, AutoModel, CLIPTextModel, CLIPTokenizer

# Hugging Face Diffusion Model库
import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, DiffusionPipeline, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.utils import convert_state_dict_to_diffusers
from diffusers.training_utils import compute_snr
from diffusers.utils.torch_utils import is_compiled_module

# 面部检测
from deepface import DeepFace

# OpenCV
import cv2
```

**参数设置**

```python
output_folder = os.path.join(project_dir, "logs") # 存放model checkpoints跟validation結果的資料夾
seed = 1126 # random seed
train_batch_size = 2 # training batch size
resolution = 512 # Image size
weight_dtype = torch.bfloat16 # 模型权重的数据类型是bfloat16
snr_gamma = 5 # Signal-to-Noise Ratio(SNR)信噪比缩放因子

## Important parameters for fine-tuning Stable Diffusion
pretrained_model_name_or_path = "stablediffusionapi/cyberrealistic-41"
lora_rank = 32 # r(rank)
lora_alpha = 16 # Lora的alpha参数

# 学习率
learning_rate = 1e-4 #@param {type:"number"}
unet_learning_rate = learning_rate
text_encoder_learning_rate = learning_rate
lr_scheduler_name = "cosine_with_restarts" # 使用Cosine Annealing with Restarts学习率调度方法
lr_warmup_steps = 100 # 初始预热步数

# 最大训练步数
max_train_steps = 200 

# 验证集数据
validation_prompt = "validation_prompt.txt"
validation_prompt_path = os.path.join(prompts_folder, validation_prompt)
validation_prompt_num = 3 #@param {type:"slider", min:1, max:5, step:1}
validation_step_ratio = 1 #@param {type:"slider", min:0, max:1, step:0.1}
with open(validation_prompt_path, "r") as f:
    validation_prompt = [line.strip() for line in f.readlines()]
```

**设置LoRA Config**

> 原版代码导入了`peft`并设置了`lora_rank`和`lora_alpha`参数，但是没有真正使用LoRA作微调。出于学习的目的，笔者改造了源代码，提供一版使用LoRA微调的代码，供大家学习交流。

Stable Diffusion模型包含三个组件：CLIP、U-net、VAE。参数量分布和占比为：
[来源](https://forums.fast.ai/t/stable-diffusion-parameter-budget-allocation/101515)

| 组件      | 参数量 | 文件大小  |  占比 |
| ----------- | ----------- |----------- |----------- |
| CLIP      | 123,060,480 |  492 MB  |  12%  |
|  VAE  | 83,653,863 |  335 MB  |  8%  |
| U-net      | 859,520,964 |  3.44 GB  |  80%  |
| Total      | 1,066,235,307 | 4.27 GB |  100%  |

U-net是最核心的组件，CLIP相对也比较重要。因此，我们选择U-net和CLIP的Attention模块进行微调。

> LoRA的核心思想是通过低秩分解降低大规模模型的参数更新需求，从而减少训练成本。其数学形式为：
> $$
> W' = W + \Delta W
> $$
> 其中：
>
> - $W$：原始权重矩阵（冻结，不更新）；
> - $\Delta W$：通过低秩分解获得的矩阵，$\Delta W = AB$
>   - $A$：低秩矩阵，形状为$(d, r)$；
>   - $B$：低秩矩阵，形状为$(r,d)$；
>   - $r$：秩，远小于$d$。
>
> 在LoRA中，直接使用低秩矩阵$AB$更新$\Delta W $有时会导致以下问题：
>
> 1. **不稳定的训练：**如果$AB$的值范围过大，与原始权重$W$相比可能导致数值不稳定，模型难以收敛；
> 2. **更新幅度过大：**如果没有约束，$AB$的影响可能过强，导致模型偏离预训练权重；
> 3. **无法灵活调节：**不同任务对参数调整的灵敏性不同，有时需要更大或更小的权重贡献。
>
> 为了解决这些问题，LoRA引入了一个缩放因子$\alpha$，控制$\Delta W$的影响：
> $$
> W' = W + \frac \alpha rAB
> $$
>
> - $\alpha$ ：缩放因子，即`lora_alpha`，用于调节$AB$对总权重更新的贡献；
> - $r$：低秩矩阵的秩，用于规范化。
>
> 通过缩放因子$\alpha$，可以有效平衡原始权重$W$和LoRA矩阵$AB$的贡献。



Step: 200 Face Similarity Score: 1.1819632053375244 CLIP Score: 30.577381134033203 Faceless Images: 0

Face Similarity Score: 1.2155983448028564 CLIP Score: 30.146756172180176 Faceless Images: 1



Step: 2000 Face Similarity Score: 1.1477864980697632 CLIP Score: 30.112869262695312 Faceless Images: 0

Face Similarity Score: 1.1696956157684326 CLIP Score: 29.713413848876954 Faceless Images: 0
