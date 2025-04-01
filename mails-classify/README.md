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

# 作业8

## Vocabulary类
1、在Vocabulary类中，mask_token对应的索引通过调用add_token方法赋值给self.mask_index属性。

2、lookup_token方法中，如果self.unk_index >=0，则对未登录词返回self.unk_index。

3、调用add_many方法添加多个token时，实际是通过循环调用add_token方法实现的。

## CBOWVectorizer类
4、vectorize方法中，当vector_length < 0时，最终向量长度等于indices的长度。

5、from_dataframe方法构建词表时，会遍历DataFrame中context和target两列的内容。

6、out_vector[len(indices):]的部分填充为self.cbow_vocab.mask_index。

## CBOWDataset类
7、_max_seq_length通过计算所有context列的token的最大值得出。

8、set_split方法通过self._lookup_dict选择对应的数据子集（DataFrame）和数据子集的长度。

9、__getitem__返回的字典中，y_target通过查找target列的token得到。

## 模型结构
10、CBOWClassifier的forward中，x_embedded_sum的计算方式是embedding(x_in).sum(dim=1)。

11、模型输出层fc1的out_features等于vocabulary_size参数的值。

## 训练流程
12、generate_batches函数通过PyTorch的DataLoader类实现批量加载。

13、训练时classifier.train()的作用是启用训练和Dropout模式。

14、反向传播前必须执行optimizer.zero_grad()清空梯度。

15、compute_accuracy中y_pred_indices通过argmax方法获取预测类别。

## 训练状态管理
16、make_train_state中early_stopping_best_val初始化为float('inf')。

17、update_train_state在连续args.early_stopping_criteria次验证损失未下降时会触发早停。

18、当验证损失下降时，early_stopping_step会被重置为0。

## 设备与随机种子
19、set_seed_everywhere中与CUDA相关的设置是torch.cuda.manual_seed_all(seed)。

20、args.device的值根据torch.cuda.is_available()确定。

## 推理与测试
21、get_closest函数中排除计算的目标词本身是通过continue判断word == target_word实现的。

22、测试集评估时一定要调用classifier.eval()方法禁用dropout。

## 关键参数
23、CBOWClassifier的padding_idx参数默认值为0。

24、args中控制词向量维度的参数是embedding_size。

25、学习率调整策略ReduceLROnPlateau的触发条件是验证损失增加（增加/减少）。