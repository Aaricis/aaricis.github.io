---
title: LangChain: Chat with Your Data
date: 2024-12-13 16:40:00 +/-8
categories: [LLM, Andrew Ng]
tags: [openai, software develop, langchain, rag]     # TAG names should always be lowercase
---

本课程是吴恩达与OpenAI、Hugging Face、LangChain等机构联合打造，面向开发者的LLM系列课程第四讲——LangChain：与你的数据对话，由LangChain联合创始人兼CEO Harrison Chase和吴恩达合作授课。

## 课程链接

[LangChain: Chat with Your Data](https://www.deeplearning.ai/short-courses/langchain-chat-with-your-data/)

>建议大家直接看DeepLearning.AI上的英文原版，配合官方提供的Jupyter Notebook效果更佳。B站上的翻译稀烂，不建议看，可能会造成误导。

## 概述

本课程主要探讨两个主题：（1）Retrieval Augmented Generation (RAG)：一种常见的LLM应用，从外部数据集中检索上下文问答；（2）Chatbot：构建一个聊天机器人，该机器人根据文档内容而不是训练中学到的资料来响应查询。

我们将探讨以下内容：

- Document Loading（文档加载）：学习数据加载的基本知识，了解LangChain提供的80多种不同的加载器，来访问包括音频和视频在内的多种数据；
- Document Splitting（文档分割）：