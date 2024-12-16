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

- Document Loading（文档加载）：学习数据加载的基本知识，了解LangChain提供的80多种不同的数据加载器，访问包括音频和视频在内的多种数据；
- Document Splitting（文档分割）：了解分割数据的最佳实践和注意事项；
- Vector stores and embeddings（向量存储和嵌入）：深入了解embedding的概念，探究在LangChain中集成向量存储的方法；
- Retrieval（检索）：掌握访问和索引向量数据的高级技术，获取比语义查询更相关的信息；
- Question Answering（问答）：建立one-pass问答解决方案；
- Chat（聊天）：使用LangChain构建Chatbot，并学习从对话和数据源中跟踪并选择相关信息；

使用LangChain和LLM构建与数据交互的实用应用程序。

## Introduction

LLM，例如ChatGPT，可以回答很多不同话题的问题，但仅在它的训练资料范围内。LLM不能与你的私有数据或者LLM训练之后出现的数据对话。本课程将要介绍如何使用LLM与你的数据对话。首先，介绍如何使用LangChain文档加载器从各种不同的数据源加载数据；然后概述语义搜索（semantic search），展示并解决其无法覆盖的边界情况；最后使用memory和LLM构建功能齐全的chatbot和你的数据对话。

## Document Loading

为了与你的数据对话，首先需要使用LangChain文档加载器把非结构化的数据处理为标准格式。LangChain提供80+种[文档加载器](https://python.langchain.com/docs/how_to/#document-loaders)，可以从不同的数据源（网站、数据库、YouTube等）加载数据，将数据转换为PDF、CSV、HTML等不同格式。

### PDFs

```python
#! pip install pypdf 

from langchain.document_loaders import PyPDFLoader
loader = PyPDFLoader("docs/cs229_lectures/MachineLearning-Lecture01.pdf")
pages = loader.load()
```

`PyPDFLoader`按页加载，每一页都是一个`Document`，包含文本（page_content）和元数据（metadata）。

```python
page = pages[0]

print(page.page_content[0:500])
```

```wiki
MachineLearning-Lecture01  
Instructor (Andrew Ng):  Okay. Good morning. Welcome to CS229, the machine 
learning class. So what I wanna do today is ju st spend a little time going over the logistics 
of the class, and then we'll start to  talk a bit about machine learning.  
By way of introduction, my name's  Andrew Ng and I'll be instru ctor for this class. And so 
I personally work in machine learning, and I' ve worked on it for about 15 years now, and 
I actually think that machine learning i
```

```python
page.metadata
```

```wiki
{'source': 'docs/cs229_lectures/MachineLearning-Lecture01.pdf', 'page': 0}
```

### YouTube

LangChain可以读取YouTube视频，使用OpenAI Whisper模型将视频转换为文本。

```python
# ! pip install yt_dlp
# ! pip install pydub

from langchain.document_loaders.generic import GenericLoader,  FileSystemBlobLoader
from langchain.document_loaders.parsers import OpenAIWhisperParser
from langchain.document_loaders.blob_loaders.youtube_audio import YoutubeAudioLoader

url="https://www.youtube.com/shorts/13c99EsNt4M"
save_dir="docs/youtube/"
loader = GenericLoader(
    YoutubeAudioLoader([url],save_dir),  # fetch from youtube
#     FileSystemBlobLoader(save_dir, glob="*.m4a"),   #fetch locally
    OpenAIWhisperParser()
)
docs = loader.load()
```

```python
docs[0].page_content[0:500]
```

```wiki
"Welcome to CS229 Machine Learning. Uh, some of you know that this is a class that's taught at Stanford for a long time. And this is often the class that, um, I most look forward to teaching each year because this is where we've helped, I think, several generations of Stanford students become experts in machine learning, got- built many of their products and services and startups that I'm sure, many of you or probably all of you are using, uh, uh, today. Um, so what I want to do today was spend s"
```

### URLs

```python
from langchain.document_loaders import WebBaseLoader

loader = WebBaseLoader("https://github.com/basecamp/handbook/blob/master/titles-for-programmers.md")
docs = loader.load()
```

### Notion

```python
from langchain.document_loaders import NotionDirectoryLoader
loader = NotionDirectoryLoader("docs/Notion_DB")
docs = loader.load()
```

## Document Splitting

![](../assets/images/llm_develop/L2-Document_splitting.png)

数据被加载为标准格式之后，被分割为更小的块（chunk）才能保存在向量数据库中。LangChain文本分割器按照chunk_size（块大小）和chunk_overlap（块间重叠大小）进行分割。

LangChain提供多种分割方式，参见[Text splitters](https://python.langchain.com/docs/concepts/text_splitters/)

### CharacterTextSplitter

[CharacterTextSplitter](https://python.langchain.com/api_reference/text_splitters/character/langchain_text_splitters.character.CharacterTextSplitter.html)基于字符分割文本。

```python
from langchain.text_splitter import CharacterTextSplitter

c_splitter = CharacterTextSplitter(
    chunk_size=26,
    chunk_overlap=4,
    separator = ' '
)

text3 = "a b c d e f g h i j k l m n o p q r s t u v w x y z"
c_splitter.split_text(text3)
```

```wiki
['a b c d e f g h i j k l m', 'l m n o p q r s t u v w x', 'w x y z']
```

> CharacterTextSplitter优先按照`separator`分割，分割后的长度可能超过`chunk_size`。

### RecursiveCharacterTextSplitter

[RecursiveCharacterTextSplitter](https://python.langchain.com/api_reference/text_splitters/character/langchain_text_splitters.character.RecursiveCharacterTextSplitter.html)是LangChain默认的文本分割器，也是推荐的文本分割器。同样基于字符分割，默认的分隔符是列表`["\n\n", "\n", " ", ""]`，以从左至右的顺序依次搜索文档中对应的分隔符进行分割。它基于文本结构将文本自然地组织成段落、句子和单词等层次单位，在拆分中保持语义一致性，适应不同级别的文本粒度。

分别用CharacterTextSplitter和RecursiveCharacterTextSplitter分割一段长文本`some_text`，`len(some_text)=496`。

```python
some_text = """When writing documents, writers will use document structure to group content. \
This can convey to the reader, which idea's are related. For example, closely related ideas \
are in sentances. Similar ideas are in paragraphs. Paragraphs form a document. \n\n  \
Paragraphs are often delimited with a carriage return or two carriage returns. \
Carriage returns are the "backslash n" you see embedded in this string. \
Sentences have a period at the end, but also, have a space.\
and words are separated by space."""
```

```python
c_splitter = CharacterTextSplitter(
    chunk_size=450,
    chunk_overlap=0,
    separator = ' '
)
c_splitter.split_text(some_text)
```

![](../assets/images/llm_develop/rag_1.png)

`some_text`在空格处被分割成长度小于chunk_size的两部分。

```python
r_splitter = RecursiveCharacterTextSplitter(
    chunk_size=450,
    chunk_overlap=0, 
    separators=["\n\n", "\n", " ", ""]
)
r_splitter.split_text(some_text)
```

![](../assets/images/llm_develop/rag_2.png)

`some_text`在'\n\n'处分割成长度小于chunk_size的两部分。

### TokenTextSplitter

[TokenTextSplitter](https://python.langchain.com/api_reference/text_splitters/base/langchain_text_splitters.base.TokenTextSplitter.html)使用tokenizer将文本分割为token。

```python
from langchain.text_splitter import TokenTextSplitter

text_splitter = TokenTextSplitter(chunk_size=1, chunk_overlap=0)
text1 = "foo bar bazzyfoo"
text_splitter.split_text(text1)
```

```wiki
['foo', ' bar', ' b', 'az', 'zy', 'foo']
```

