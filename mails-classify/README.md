# 代码核心功能说明

## 算法基础
算法基础：采用多项式朴素贝叶斯分类器,通过独立性假设简化联合概率计算，结合贝叶斯定理将邮件分类问题转化为概率比较问题,具体形式为： 
  $$P(y|x) = \frac{P(x|y) \cdot P(y)}{P(x)}$$
  其中：
  - $P(y|x)$是邮件内容 $x$ 属于类别 $y$ 的后验概率。
  - $P(x|y)$ 是类别 $y$ 下邮件内容 $x$ 的似然概率。
  - $P(y)$ 是类别 $y$ 的先验概率。
  - $P(x)$ 是邮件内容 $x$ 的边际概率。

特征独立性假设
在多项式朴素贝叶斯中，假设每个特征（即每个单词）在给定类别下是相互独立的。也就是说，一个邮件中某个单词的出现与否，不会影响其他单词的出现概率。例如，在判断一封邮件是否为垃圾邮件时，“优惠” 这个词的出现与 “抽奖” 这个词的出现是相互独立的事件。
数学表达为：
  $$P(x|y) = \prod_{i=1}^{n} P(x_i|y)$$
  
  其中 $x_i$ 表示邮件中的第 $i$ 个词。 

## 数据处理流程

分词处理

![img.png](img.png)

这行代码将每一行文本分割成一个个单词。

## 停用词过滤

通过正则表达式过滤了无效字符，并过滤了长度为 1 的词。

![img_1.png](img_1.png)

这样可以去除一些没有实际意义的字符和单字，减少噪声。

## 特征构建过程

### 高频词特征选择

高频词特征选择是基于词频的方法。对于每一封邮件，统计每个高频词在邮件中出现的次数，将这些次数作为特征向量。

#### 数学表达形式

假定我们有一个邮件集合 $D$  ，其中包含 $n$ 封邮件，即 $D = \{d_1, d_2, \cdots, d_n\}$ 。
对于每一封邮件 $d_i(i = 1, 2, \cdots, n)$  ，我们挑选 $k$ 个高频词作为特征。对于第 $i$ 封邮件 $d_i$  ，
其特征向量表示为 $X_i = (x_{i1}, x_{i2}, \cdots, x_{ik})$  。这里 $x_{ij}$ 代表第 $j$ 个高频词在第 $i$ 封
邮件 $d_i$ 中出现的次数。
### TF-IDF 特征加权

TF-IDF（Term Frequency - Inverse Document Frequency）是一种用于信息检索与文本挖掘的常用加权技术。TF-IDF
值由两部分组成：词频（TF）和逆文档频率（IDF）。

#### 数学表达形式

词频（TF）表示某个词w在一篇文档d中出现的频率。计算公式为：
 $$TF(w, d)=\frac{count(w, d)}{\sum_{w' \in d}count(w', d)}$$ 
其中， $count(w, d)$  表示词w在文档d中出现的次数， $\sum_{w' \in d}count(w', d)$  表示文档d中所有词出现的次数之和。

逆文档频率（IDF）表示某个词w在整个文档集合D中的普遍重要性。计算公式为：
$$IDF(w)=\log\frac{|D|}{|\{d \in D: w \in d\}|}$$
其中， $|D|$ 表示文档集合D中的文档总数， $|\{d \in D: w \in d\}|$ 表示包含词w的文档数。

TF-IDF 值：
一个词w在文档d中的 TF-IDF 值是词频（TF）和逆文档频率（IDF）的乘积，即：
$$TF - IDF(w,d)=TF(w,d)\times IDF(w)$$

#### 两种方法的差异

高频词特征选择只考虑了词在文档中的出现次数，而没有考虑词在整个文档集合中的普遍重要性。因此，一些常见的词（如 “的”、“是”
等）可能会在特征向量中占据较大的比重，影响分类效果。

TF-IDF 特征加权则综合考虑了词在文档中的出现频率和在整个文档集合中的普遍重要性。通过 IDF 的加权，能够降低常见词的影响，提高分类的准确性。

## 两种特征模式的切换方法

![img_2.png](img_2.png)

切换单引号中的字符进行两种特征模式的切换。

## 运行截图

### 高词频模式运行截图

![img_3.png](img_3.png)

### TF-IDF模式运行截图

![img_4.png](img_4.png)

### 任务四运行截图

![img_5.png](img_5.png)

### 任务五运行截图

![img_6.png](img_6.png)

