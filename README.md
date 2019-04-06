# BaseTensorFlowModel
## 模型简介：
>在这个仓库当中本人旨实现深度学习的一些常见的模型：CNN、LSTM、GRU，然后再这些baseline模型中添加不同的注意力，在MR数据集当中验证结果。
## 实验数据集
实验数据集：
下载地址：https://www.cs.cornell.edu/people/pabo/movie-review-data/

### 数据划分方式
Vocabulary Size: 18772

Train/Dev split: 9596/1066


## 实验模型和结果：
### CNN
#### CNN部分我采用的是Kim经典文章里的模型：

![upload successful](\\images\pasted-4.png\)

>论文下载地址：https://arxiv.org/pdf/1408.5882

目的是为了对比kim这篇文章的结果：

![upload successful](\\images\pasted-5.png\)

#### CNN+Att中的注意力部分
受到实验室师兄的启发，我试着在卷积上实现了一下注意力机制：



>论文下载地址：https://pdfs.semanticscholar.org/4946/89f4522619b887e515aea2b205490b0eb5cd.pdf

#### CNN部分的实验结果


![upload successful](\\images\pasted-3.png\)

### RNN
> RNN部分Baseline是LSTM（or双向）、GRU（or双向），并在两者上面添加了注意力机制

#### 注意力模型图

用手画的，有点low：


#### RNN部分实验结果



### 问题

* 1.为何loss不降反升但acc还是不变？

* 2.矩阵3维*2维如何做到？

* 3.None的问题？

* 4.第一次尝试复现别人文章里的一些东西，理解是否有误？



