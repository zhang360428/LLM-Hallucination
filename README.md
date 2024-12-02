# LLM-Hallucination
Triggering and alleviating hallucinations in large language models
本项目主要是在大模型幻觉触发与缓解方面的探索。并且提供了一个简单的可替换的框架，让使用者只需要自己替换main.py中的幻觉触发与缓解函数能对大模型幻觉有简单的了解。

输入在Triggering_Hallucinations和Alleviating_Hallucinations两个文件夹下的Input的Q.txt中，读者可自行替换为自己想要触发幻觉的问题，相应的输出会生成在两个文件夹下的Output中

## 大模型幻觉触发
读者可以通过替换perturb_text函数，将幻觉触发函数更改为自己的函数
在该部分我们期望在不改变原始问题语义的前提下，通过对问题文本的扰动重构，尽可能提高目标大模型生成幻觉的概率。
1. 文本扰动与重构
为了增强问题的多样性和混淆性，但同时不显著改变其语义，对输入问题 Q进行了如下操作：
同义词替换：使用 NLP 工具（如 spaCy 和 WordNet），根据词性精准替换句子中的核心词汇，确保生成的句子在语义上尽量不变。
句子结构扰动：基于句法分析，提取句子中的子句并随机重排，使句子结构发生改变。
2. 插入无关信息



## 大模型幻觉缓解
读者可以通过替换Mitigate_hallucination函数将幻觉缓解函数改为自己的函数，并且除了将Q.txt更改为自己的问题以为，还可以将KB.txt替换为自己的检索知识库
