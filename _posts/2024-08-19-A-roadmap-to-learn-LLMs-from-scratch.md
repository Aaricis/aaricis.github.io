# <span style="color: orange;">前言</span>
## <span style="color: blue;">维度与层次</span>
AI大模型技术雷达图从能力角度分为以下几个维度：
- 大模型基本概念和原理
- 模型架构
- 大模型高阶应用
- 大模型实战
- 算力
针对这几个维度中众多的技术点，有偏重地推荐为两个层次：
<span style="color: blue;">**关键知识技能：** </span>表格中标蓝加粗字体为关键的知识技能与技术点；
<span style="color: blue;">**扩展知识技能：**</span> 未标蓝加粗的内容为推荐的扩展内容，在掌握了关键知识技能的基础上，根据业务的不同可以有选择性地挑选学习；
## <span style="color: blue;">雷达图的应用建议</span>
- <span style="color: blue;">**知道(1)分**</span>：能说清楚是什么,解决什么问题,了解技术对应的社区的使用情况和学习路径。
- <span style="color: blue;">**会用(2)分：**</span>实现过对应技术的"QuickStart",知道技术的适用场景,能照猫画虎实现需求。
- <span style="color: blue;">**熟练(3)分：**</span>能够在业务场景中用最佳实践解决问题,形成自己的方法论和套路。
- <span style="color: blue;">**掌握(4)分：**</span>熟悉技术背后原理,研究过源码,能够解决疑难问题(故障、性能优化以及扩展)。

# <span style="color: orange;">大模型基本概念和原理</span>

| 知识技能      | 技术点                               | 知道                                                         | 会用                                                         | 熟练                                                         | 掌握                                                         |
| ------------- | ------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **前置学习**  | **机器学习（Machine Learning，ML）** | [机器学习](https://www.showmeai.tech/article-detail/185)     | [吴恩达机器学习](https://www.bilibili.com/video/BV164411b7dx/?from=search&seid=5856176897296408924) |                                                              |                                                              |
|               | **深度学习（Deep Learning，DL）**    | [神经网络](https://www.elastic.co/cn/what-is/neural-network)、[深度学习简介](https://github.com/datawhalechina/leedl-tutorial) |                                                              |                                                              |                                                              |
| **LLM基础**   | **AI技术词扫盲**                     | [人工智能](https://zh.wikipedia.org/wiki/人工智能)、[AGI](https://zh.wikipedia.org/wiki/通用人工智慧)、[AIGC](https://zh.wikipedia.org/wiki/生成式人工智慧)、[Large language model](https://zh.wikipedia.org/wiki/大型语言模型)、[NLP](https://zh.wikipedia.org/zh-hans/NLP) | [Prompt engineering](https://zh.wikipedia.org/wiki/生成式人工智慧)、[过拟合、欠拟合](https://www.hrwhisper.me/machine-learning-regularization/) | [fine-tuning](https://platform.openai.com/docs/guides/fine-tuning/fine-tuning)、[In Context Learning, ICL](https://www.hopsworks.ai/dictionary/in-context-learning-icl)、[Chain-of-Thought, CoT](https://www.promptingguide.ai/techniques/cot) | [RLHF](https://zh.wikipedia.org/wiki/基于人类反馈的强化学习) |
|               | **大模型简介和技术发展趋势**         | [机器学习、深度学习发展历史](https://3ms.huawei.com/km/groups/2033815/blogs/details/14718863?l=zh-cn&moduleId=407394802882781184) | [生成式AI](https://speech.ee.ntu.edu.tw/~hylee/genai/2024-spring.php) | [大语言模型简介](https://ilearning.huawei.com/course/100000/application-learn/CNE202307141429004-478?classCode=&sourcesType=&sxz-lang=zh_CN) |                                                              |
|               | **人工智能与大模型基础**             | [人工智能基础](https://ilearning.huawei.com/course/100000/application-learn/COU20240531000016?classCode=&sourcesType=&sxz-lang=zh_CN#323fdee14a0744c8a572a4c81c55bb2e) | [大模型科普&电信领域实践赋能系列课程](https://ilearning.huawei.com/edx/next/program/559746268793802752) | [大模型之美](https://ilearning.huawei.com/edx/next/courses-learn?courseId=100541001&articleId=644795) |                                                              |
| **LLM提示词** | **prompt**                           | [learnprompting](https://learnprompting.org/zh-Hans/docs/intro)、[提示工程指南](https://www.youtube.com/watch?v=yhk9x__D-Us) | [如何写好prompt](https://ilearning.huawei.com/course/100000/application-learn/COU20240527000015?classCode=&sourcesType=&sxz-lang=zh_CN#ebedd9a057c540acbeaa495b15609bae)、[ChatGPT 中文调教指南](https://github.com/PlexPt/awesome-chatgpt-prompts-zh)、[prompt examples](https://platform.openai.com/docs/examples) | [openai prompt engneering](https://platform.openai.com/docs/guides/prompt-engineering/prompt-engineering) | [吴恩达教你写提示词](https://www.youtube.com/watch?v=gCbHoXL2IcA&list=PL1bD7CLmGebf9FewSIJGZhFDK6FyWBH7P) |
| **算法原理**  | **机器学习算法**                     | [机器学习算法原理](https://speech.ee.ntu.edu.tw/~hylee/ml/2022-spring.php)、[KNN](https://www.showmeai.tech/article-detail/187)、[逻辑回归](https://www.showmeai.tech/article-detail/188)、[朴素贝叶斯](https://www.showmeai.tech/article-detail/189)、[决策树](https://www.showmeai.tech/article-detail/190) | [随机森林](https://www.showmeai.tech/article-detail/191)、[回归树](https://www.showmeai.tech/article-detail/192)、[支持向量机](https://www.showmeai.tech/article-detail/196)、[聚类算法](https://www.showmeai.tech/article-detail/197)、[PCA降维](https://www.showmeai.tech/article-detail/198) | [GBDT](https://www.showmeai.tech/article-detail/193)、[LightGBM](https://www.showmeai.tech/article-detail/195)、[XGBoost](https://www.showmeai.tech/article-detail/194) |                                                              |
|               | **深度学习算法**                     | [深度学习简介](https://speech.ee.ntu.edu.tw/~hylee/ml/2022-spring.php) | [梯度下降](https://www.showmeai.tech/article-detail/217)、[损失函数](https://www.showmeai.tech/article-detail/262)、[反向传播](https://www.showmeai.tech/article-detail/234) | [Transformers自注意力](https://www.showmeai.tech/article-detail/251)、[Transformer](https://www.youtube.com/watch?v=n9TlOhRjYoc) | [迁移学习算法](https://ilearning.huawei.com/course/100000/application-learn/CNE202110291211015-875?classCode=&sourcesType=&sxz-lang=zh_CN#e5c43474772d4606adb1f7b34edb7b8e)、[联邦学习](https://ilearning.huawei.com/edx/next/programs/cf5fa8b4-f047-4b1e-b9a2-c9cb233a4313/about)、[《深度学习》中的线性代数基础](https://www.jiqizhixin.com/articles/boost-data-science-skills-learn-linear-algebra) |





