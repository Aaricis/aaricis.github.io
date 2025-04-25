---
title: 李宏毅-ML2022-HW7-BERT
date: 2025-04-14 15:07:00 +/-8
categories: [Machine Learning 2022]
tags: [Hung-yi Lee]     # TAG names should always be lowercase
---

## Task Introduction

使用Bert模型进行抽取式问答（Extractive Question Answering）。输入文本和问题，返回答案在文本中开始和结束的位置。

## Dataset

DRCD: 台達閱讀理解資料集 Delta Reading Comprehension Dataset；

ODSQA: Open-Domain Spoken Question Answering Dataset；

- train: DRCD + DRCD-TTS
  - 10524 paragraphs, 31690 questions 
- dev: DRCD + DRCD-TTS 
  - 1490 paragraphs, 4131 questions 
- test: DRCD + ODSQA
  - 1586 paragraphs, 4957 questions

## 思路

### Simple Baseline(0.45139)

Score: 0.47882

Private score: 0.46569

跑一遍Sample Code。

### Medium Baseline(0.65792)

Score: 0.69302

Private score: 0.68684

- Apply linear learning rate decay;
  
  根据助教提示，使用带warm up 的 learning rate scheduler。

  ```python
  from transformers import get_linear_schedule_with_warmup
  
  # total training steps
  total_steps = len(train_loader) * num_epoch
  num_warmup_steps = int(0 * total_steps)  # Set warmup steps to 20% of total steps
  
  # [Hugging Face] Apply linear learning rate decay with warmup
  scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=total_steps
  )
  ```

- Change value of `doc_stride`;
  
  `doc_stride`表示两个连续窗口的起始位置之间的距离。默认值为`max_paragraph_len`，此时窗口是不重叠的，也就是说第一个窗口为[0, 149]，第二个窗口为[150, 299]......如果答案位于[140, 160]，默认的设置无法捕捉到答案。因此，需要调整`doc_stride`，使窗口之间发生重叠。`doc_stride`可以理解为截取文本时，窗口每次滑动的步长。

  `doc_stride`只在验证和测试阶段使用，与训练无关。经测试，`doc_stride`取`max_paragraph_len * 0.25`。
  ```python
  self.doc_stride = int(self.max_paragraph_len * 0.25)
  ```

### Strong Baseline(0.78136)
Score: 0.79548

Private score: 0.78974

- Improve preprocessing
  
  Sample Code以答案为中心截取训练文本，会让模型误以为答案都在文本的中心。我们增加随机偏移，让答案不总是在文本中心。

  ```python
  # 防止模型学习到「答案总是位于中间的位置」，加入随机偏移
  max_offset = self.max_paragraph_len // 2   # 最大偏移量为段落长度的1/2，这是可调的
  random_offset = np.random.randint(-max_offset, max_offset)  # 在 [-max_offset, +max_offset] 范围内随机选择偏移量
  paragraph_start = max(0, min(mid + random_offset - self.max_paragraph_len // 2, len(tokenized_paragraph) - self.max_paragraph_len))
  paragraph_end = paragraph_start + self.max_paragraph_len
  ```

- Try other pretrained models

  尝试不同的预训练模型，"hfl/chinese-roberta-wwm-ext-large"更强的RoBERTa中文WWM（Whole Word Masking，全词掩码模型）,表现好于Bert。
  
  >WWM（Whole Word Masking，全词掩码）是BERT及其变种模型中的一种预训练技术，主要用于改进中文（以及类似语言）的掩码语言建模（MLM）任务。它是针对原始BERT的字级别掩码（Character-level Masking）的优化方案。

  ```python
  model = BertForQuestionAnswering.from_pretrained("hfl/chinese-roberta-wwm-ext-large").to(device)
  tokenizer = BertTokenizerFast.from_pretrained("hfl/chinese-roberta-wwm-ext-large")
  ```

### Boss Baseline(0.84388)

Score: 0.84630

Private score: 0.83857

- Improve postprocessing

  Sample Code `def evaluate()`会出现`start_index`大于`end_index`的情况，导致`[start_index : end_index + 1]`无法捕获到答案。查看 result.csv 文件时，可以发现有些结果是空的，我们需要修正这个问题。
  - (1)只考虑`start_index < end_index`的区间，选择概率总和最大的区间。
  Score: 0.78660 Private score: 0.79217
  ```python
  def evaluate(data, output):
    answer = ''
    max_prob = float('-inf')
    num_of_windows = data[0].shape[1]
  
    for k in range(num_of_windows):
        start_logits = output.start_logits[k]  # shape: (seq_len,)
        end_logits = output.end_logits[k]     # shape: (seq_len,)
        
        # 向量化计算所有组合的概率和 (seq_len, seq_len)
        prob_matrix = start_logits.unsqueeze(1) + end_logits.unsqueeze(0)
        
        # 生成上三角掩码（确保end >= start）
        mask = torch.triu(torch.ones_like(prob_matrix, dtype=torch.bool))
        prob_matrix = prob_matrix.masked_fill(~mask, float('-inf'))
        
        # 找到最大概率的合法组合
        best_prob, best_idx = torch.max(prob_matrix.flatten(), dim=0)
        best_start, best_end = np.unravel_index(best_idx.item(), prob_matrix.shape)
  
        if best_prob > max_prob:
            max_prob = best_prob
            answer = tokenizer.decode(data[0][0][k][best_start : best_end + 1])
  
    return answer.replace(' ', '')
  ```
  - (2)直接跳过非法区间。
  
    Score: 0.78217  Private score: 0.78894
    ```python
    def evaluate(data, output):
        answer = ''
        max_prob = float('-inf')
        num_of_windows = data[0].shape[1]
    
        for k in range(num_of_windows):
            start_prob, start_index = torch.max(output.start_logits[k], dim=0)
            end_prob, end_index = torch.max(output.end_logits[k], dim=0)
    
            # 跳过非法区间
            if end_index < start_index:
                continue
    
            prob = start_prob + end_prob
    
            if prob > max_prob:
                max_prob = prob
                answer = tokenizer.decode(data[0][0][k][start_index : end_index + 1])
    
        return answer.replace(' ', '')
    ```
  - (3)枚举所有合法的区间，找出最大概率的那一对，并限制答案长度不超过`max_answer_length=30`。
  
  Score: 0.80072 Private score: 0.79782
  ```python
  def evaluate(data, output):
    answer = ''
    max_prob = float('-inf')
    num_of_windows = data[0].shape[1]
    max_answer_length = 30
  
    for k in range(num_of_windows):
        start_logits = output.start_logits[k]   # shape: (seq_len,)
        end_logits = output.end_logits[k]       # shape: (seq_len,)
  
        seq_len = start_logits.size(0)
  
        # 构造得分矩阵 score[i][j] = start_logits[i] + end_logits[j]
        start_logits = start_logits.unsqueeze(1)         # (seq_len, 1)
        end_logits = end_logits.unsqueeze(0)             # (1, seq_len)
        score_matrix = start_logits + end_logits         # (seq_len, seq_len)
  
        # 创建 mask：只保留满足 end >= start 且 长度 <= max_answer_length 的组合
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=0)  # end >= start
        mask = mask * torch.tril(torch.ones(seq_len, seq_len), diagonal=max_answer_length - 1)  # 限制长度
        mask = mask.to(score_matrix.device)
  
        # 将非法位置的分数设为极小值
        score_matrix = score_matrix.masked_fill(mask == 0, float('-inf'))
  
        # 找到最大得分及其索引
        flat_index = torch.argmax(score_matrix)
        start_index = flat_index // seq_len
        end_index = flat_index % seq_len
  
        prob = score_matrix[start_index, end_index]
        if prob > max_prob:
            max_prob = prob
            answer = tokenizer.decode(data[0][0][k][start_index:end_index + 1])
  
    return answer.replace(' ', '')
  ```

  综上，只有第三版`def evaluate()`Score略超过Strong Baseline，后续实验都使用此方法作推理。
- 梯度累积，即每n个step更新一次梯度，相当于将batchsize扩大n倍。
  使用梯度累积将batchsize扩大为64。
  ```python
  gradient_accumulation_steps = 2
  ```
  Score: 0.81202 Private score: 0.

- Adjust learning rate automatically by scheduler。
  使用`get_cosine_schedule_with_warmup`，warmup step改为`1.15 * 0.1 * total_steps`。
    ```python
    # total training steps
    total_steps = len(train_loader) * num_epoch
    num_warmup_steps= int(1.15 * 0.1 * total_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=total_steps)
    ```
- 其他
降低学习率到`1e-5`，增加epoch=4

- Ensemble多次尝试的结果。

## Code

[双过Boss Baseline](https://github.com/Aaricis/Hung-yi-Lee-ML2022/tree/main/HW7)

## Report

1. **(2%) After your model predicts the probability of answer span start/end position, what rules did you apply to determine the final start/end position? (the rules you applied must be different from the sample code)** 

   - 枚举所有合法的区间，找出最大概率的那一对，并限制答案长度不超过`max_answer_length=30`；

   - 如果answer中出现`[UNK]`，说明有某些字符无法正常编码解码，因此将`[UNK]`还原为原始文本。

     


2.  **(2%) Try another type of pretrained model which can be found in huggingface’s Model Hub (e.g. BERT -> BERT-wwm-ext, or BERT -> RoBERTa ), and describe**

   ● the pretrained model you used 

   `luhua/chinese_pretrain_mrc_macbert_large`

   ● performance of the pretrained model you used 
   
   Ensemble: Score: 0.84630 Private score: 0.83857
   
   Single: Score: 0.83662 Private score: 0.83535
   
   ● the difference between BERT and the pretrained model you used (architecture, pretraining loss, etc.)
   
   `luhua/chinese_pretrain_mrc_macbert_large` 是一个 **中文问答任务专用的预训练模型**，它基于 [MacBERT](https://github.com/ymcui/MacBERT) 预训练架构，专门在中文机器阅读理解（MRC）数据集上进行过进一步的预训练。其在预训练时**将词替换为相似词**（disentangled MLM），而不是token；更强调**语义理解**，更适合下游问答、抽取式任务。

## Reference

[李宏毅机器学习 hw7 boss baseline分享_2023李宏毅hw7-CSDN博客](https://blog.csdn.net/qq_43613342/article/details/127044475)

[BERT（李宏毅）机器学习 2023 Spring HW7 - 知乎](https://zhuanlan.zhihu.com/p/11697603079)

[李宏毅2022机器学习HW7解析 - 知乎](https://zhuanlan.zhihu.com/p/516095759)
