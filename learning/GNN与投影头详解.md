# GraphCheck 中 GNN 与投影头的详细说明

这份文档专门回答一个问题：

> 在 GraphCheck 里，GNN 和投影头到底分别做了什么？它们为什么要这样设计？训练时又是怎样被优化的？

如果只用一句话概括当前实现，可以写成：

> GraphCheck 先把 claim 图和 document 图分别编码成两个图级向量，再用投影头把这两个图向量映射到 LLM 词向量空间，最后把它们当作两个额外的条件位置拼进冻结 LLM 的输入 embedding 序列里，让 LLM 在文本判断任务中利用图结构信息。

---

## 1. 整体位置：GNN 和投影头在整个模型里处于哪一层

GraphCheck 的主体结构在 `model/graphcheck.py` 中定义。

模型初始化时，核心模块有三类：

1. 一个冻结的因果语言模型 LLM；
2. 一个可训练的图编码器 `graph_encoder`；
3. 一个可训练的投影头 `projector`。

对应代码见 `model/graphcheck.py:51`、`model/graphcheck.py:73`、`model/graphcheck.py:83`。

它们之间的关系可以先看成下面这条链路：

```text
claim_kg/doc_kg
    -> GNN
    -> 图级向量
    -> projector
    -> LLM embedding 空间中的两个向量
    -> 拼接到文本 embedding 前面
    -> 冻结 LLM
    -> 生成 support / unsupport
```

因此：

- GNN 的职责是把图结构压缩成可用的图表示；
- 投影头的职责是把图表示变成 LLM 吃得下的表示；
- 冻结 LLM 的职责是把文本条件和图条件整合起来，生成最终标签。

---

## 2. 图数据最初长什么样

要理解 GNN 在做什么，先要知道输入给 GNN 的图不是原始三元组字符串，而是已经预处理成了 PyTorch Geometric 的 `Data` 对象。

这一步发生在 `graph_build.py`。

### 2.1 原始图来源

数据集里每条样本都有两份知识图：

- `claim_kg`：由 claim 抽取出来的图；
- `doc_kg`：由 document 抽取出来的图。

在 `dataset/utils/dataset.py:54` 到 `dataset/utils/dataset.py:59` 中，可以看到每条样本会同时拿到这两张图。

### 2.2 三元组先被转成节点表和边表

`graph_build.py:35` 中的 `textualize_graph()` 会把图三元组：

```text
(head, relation, tail)
```

转成：

- 节点表 `nodes`：每个节点有一个文本属性 `node_attr` 和一个整数 `node_id`
- 边表 `edges`：每条边有 `src`、`edge_attr`、`dst`

也就是说，当前项目里：

- 节点特征来自实体文本；
- 边特征来自关系文本。

### 2.3 节点文本和边文本被编码成稠密向量

在 `graph_build.py:158` 到 `graph_build.py:184` 中：

- `x = text2embedding(..., nodes.node_attr.tolist())` 把节点文本编码为节点特征矩阵；
- `e = text2embedding(..., edges.edge_attr.tolist())` 把关系文本编码为边特征矩阵。

这个 `text2embedding` 来自 `dataset/utils/modeling.py:70`。

它内部使用了 `sentence-transformers/all-roberta-large-v1`，先得到文本 token 表示，再做 mean pooling，并做 L2 归一化，见 `dataset/utils/modeling.py:33` 到 `dataset/utils/modeling.py:51`。

所以在图进入 GNN 之前：

- 每个节点已经是一个 1024 维文本语义向量；
- 每条边也已经是一个 1024 维关系语义向量。

这也是为什么配置中 `gnn_in_dim=1024`，见 `src/config.py:48`。

---

## 3. GNN 的直接输入和输出分别是什么

训练时，`GraphCheck.encode_graphs()` 会取出 batch 中的两类图：

```python
claim_kg = data["claim_kg"].to(self.model.device)
doc_kg = data["doc_kg"].to(self.model.device)
```

见 `model/graphcheck.py:106` 到 `model/graphcheck.py:107`。

随后调用：

```python
claim_n_embeds, _ = self.graph_encoder(
    claim_kg.x, claim_kg.edge_index.long(), claim_kg.edge_attr
)
doc_n_embeds, _ = self.graph_encoder(
    doc_kg.x, doc_kg.edge_index.long(), doc_kg.edge_attr
)
```

见 `model/graphcheck.py:110` 到 `model/graphcheck.py:115`。

因此，GNN 的输入是：

- `x`：节点特征矩阵，形状近似为 `[num_nodes, 1024]`
- `edge_index`：图连接关系，形状近似为 `[2, num_edges]`
- `edge_attr`：边特征矩阵，形状近似为 `[num_edges, 1024]`

GNN 的直接输出是：

- 每个节点更新后的节点表示 `node_embeddings`

注意：GNN 此时输出的还不是最终给 LLM 的那两个向量，而是节点级表示。后面还要做图级池化和投影。

---

## 4. 当前项目提供了哪两种 GNN

在 `model/gnn.py:129` 到 `model/gnn.py:133` 中，项目通过名字映射来选择具体图编码器：

```python
load_gnn_model = {
    "gat": GAT,
    "gt": GraphTransformer,
}
```

默认配置是：

```python
parser.add_argument("--gnn_model_name", type=str, default="gat")
```

见 `src/config.py:46`。

也就是说，如果不额外改参数，训练时默认使用的是 `GAT`。

下面把两种实现分别拆开解释。

---

## 5. `GAT` 在这个项目里具体做了什么

`GAT` 定义在 `model/gnn.py:69`。

### 5.1 网络结构

它由多层 `GATConv` 组成：

- 第一层：`in_channels -> hidden_channels`
- 中间若干层：`hidden_channels -> hidden_channels`
- 最后一层：`hidden_channels -> out_channels`

每层之间基本遵循：

```text
GATConv -> BatchNorm -> ReLU -> Dropout
```

最后一层不再做激活和 dropout，只输出最终节点表示。

对应代码见 `model/gnn.py:82` 到 `model/gnn.py:95`、`model/gnn.py:104` 到 `model/gnn.py:126`。

### 5.2 它处理了什么信息

`GATConv` 的核心思想是：

- 一个节点更新自己时，不是对邻居简单平均；
- 而是给不同邻居分配不同注意力权重；
- 重要邻居对当前节点的影响更大，不重要邻居影响更小。

所以在这个任务里，GAT 的实际作用是：

- 根据图结构决定哪些邻居信息应该被强调；
- 将节点自己的实体语义与邻居传播来的关系上下文融合起来；
- 逐层得到更适合判别 claim/doc 语义关系的节点表示。

### 5.3 多头注意力在这里怎么用

代码里是：

```python
GATConv(..., heads=num_heads, concat=False)
```

见 `model/gnn.py:83`、`model/gnn.py:89`、`model/gnn.py:93`。

这说明：

- 每层会并行计算多个注意力头；
- 但最后不把各头结果拼接起来，而是合成一个固定维度输出；
- 这样可以让层与层之间维度保持稳定，便于堆叠。

### 5.4 它还保留了注意力权重

`GAT.forward()` 中调用了：

```python
return_attention_weights=True
```

并把最后一层注意力权重保存到：

```python
self.attn_weights = attn_weights_list[-1]
```

见 `model/gnn.py:108` 到 `model/gnn.py:124`。

这说明当前实现虽然训练主流程只用节点表示，但也给后续分析留了接口。你如果想研究“哪些边在图推理里更重要”，可以从这里入手做可视化或解释性分析。

---

## 6. `GraphTransformer` 在这个项目里具体做了什么

`GraphTransformer` 定义在 `model/gnn.py:6`。

### 6.1 网络结构

它使用的是 `TransformerConv`，总体结构和 `GAT` 很像：

```text
TransformerConv -> BatchNorm -> ReLU -> Dropout
```

最后一层只输出表示，不再激活。

对应代码见 `model/gnn.py:17` 到 `model/gnn.py:50`、`model/gnn.py:58` 到 `model/gnn.py:66`。

### 6.2 它和 GAT 的主要区别

当前代码中，`TransformerConv` 的一个显式特点是：

```python
edge_dim=in_channels
```

见 `model/gnn.py:24`、`model/gnn.py:36`、`model/gnn.py:46`。

这说明边特征会被直接纳入消息传递过程。

因此，和普通只看邻居节点的图层相比，它更强调：

- 节点是什么；
- 邻居是什么；
- 节点和邻居之间的关系文本又是什么。

换句话说，在知识图场景里，`GraphTransformer` 比较适合表达：

> “谁和谁之间通过哪种关系连接”

而不仅是：

> “谁和谁连接了”

### 6.3 它在本项目里的直观作用

如果 claim 图或 doc 图中，某些关系词本身就非常关键，例如“causes”“located in”“denies”“supports”之类，那么显式利用边特征的图层会更有机会把这类关系信号编码进节点表示中。

---

## 7. GNN 输出以后，为什么还要做图级池化

在 `model/graphcheck.py:117` 到 `model/graphcheck.py:139` 中，节点表示会进一步做平均池化：

```python
claim_embeds = scatter(claim_n_embeds, claim_kg.batch, dim=0, reduce="mean")
doc_embeds = scatter(doc_n_embeds, doc_kg.batch, dim=0, reduce="mean")
```

如果不是 batch 图，则退化为：

```python
claim_embeds = claim_n_embeds.mean(dim=0, keepdim=True)
doc_embeds = doc_n_embeds.mean(dim=0, keepdim=True)
```

### 7.1 这一层在做什么

它把：

- 一个样本中若干个 claim 节点表示
- 一个样本中若干个 doc 节点表示

分别压缩成：

- 1 个 claim 图向量
- 1 个 doc 图向量

### 7.2 为什么要这么做

因为当前项目并没有把整张图展开成很多“图 token”送给 LLM，而是只给 LLM 两个额外条件位置：

- 一个位置放 claim 图摘要；
- 一个位置放 doc 图摘要。

所以这一步的意义是：

- 把图从节点级表示压缩成图级摘要；
- 控制序列长度，不让图节点数直接膨胀 LLM 上下文长度；
- 让每条样本固定只注入两个图条件位置，训练和推理都更稳定。

### 7.3 这个设计的优点和代价

优点：

- 简洁；
- 序列长度开销非常低；
- 容易和大模型输入拼接；
- 即便图大小差异很大，也能统一成固定数量的向量。

代价：

- 图内细粒度节点差异会被平均掉一部分；
- LLM 看不到每个节点或每条边的局部细节；
- 所有图信息都被压缩进两个摘要向量里，信息瓶颈比较明显。

因此，当前实现属于一种很明确的“图摘要注入”路线，而不是“节点级图 token 注入”路线。

---

## 8. 投影头 `projector` 为什么必须存在

在 `model/graphcheck.py:83` 到 `model/graphcheck.py:87` 中：

```python
self.projector = nn.Sequential(
    nn.Linear(args.gnn_hidden_dim, 2048),
    nn.Sigmoid(),
    nn.Linear(2048, self.word_embedding.weight.shape[1]),
)
```

这说明投影头是一个两层 MLP：

```text
gnn_hidden_dim -> 2048 -> llm_embed_dim
```

中间带一个 `Sigmoid` 非线性。

### 8.1 它最直接的作用：解决维度不匹配

GNN 输出的是图向量，维度默认是：

```text
gnn_hidden_dim = 1024
```

而 LLM 的词向量维度是：

```python
self.word_embedding.weight.shape[1]
```

也就是底层 LLM 的 embedding 维度，通常远大于或不同于 1024。

如果没有投影头，这两个空间根本无法直接拼接到同一条 `inputs_embeds` 序列里。

因此，投影头首先解决的是最基本的接口问题：

> 把 GNN 输出变成和 LLM token embedding 同维度的向量。

### 8.2 它更深层的作用：完成语义空间对齐

维度对齐只是表面功能，更重要的是语义对齐。

GNN 输出向量所在的空间，本质上是：

- 由图结构消息传递学出来的表示空间；
- 用于表达节点和图在当前任务中的结构语义。

而 LLM 词向量空间是：

- 预训练语言模型长期学出来的 token 语义空间；
- 用于后续自注意力和生成过程。

这两个空间即使维度恰好一样，也不意味着它们可以直接互通。

投影头的真正价值在于：

> 学会把“图语义摘要”翻译成“冻结 LLM 能理解的 embedding 形式”。

你可以把它看成一个小型适配器，作用类似“跨模态翻译器”。

### 8.3 为什么是两层 MLP，而不是一层线性层

当前实现没有只用：

```python
nn.Linear(gnn_hidden_dim, llm_embed_dim)
```

而是用了两层加非线性：

```text
Linear -> Sigmoid -> Linear
```

这样做的潜在原因是：

- 给投影头更强的表示能力；
- 允许它学习非线性的空间变换；
- 让图向量更灵活地贴近 LLM embedding 分布。

当然，`Sigmoid` 也带来一个值得注意的点：

- 它会把中间层压到有限区间；
- 可能有助于稳定，但也可能带来一定饱和风险；
- 这属于当前实现的设计选择，不一定是唯一或最优解。

---

## 9. 投影后的两个图向量到底被放到哪里

在 `model/graphcheck.py:167` 到 `model/graphcheck.py:170` 中：

```python
claim_embeds = self.projector(claim_embeds)
doc_embeds = self.projector(doc_embeds)
```

随后在 `model/graphcheck.py:220` 到 `model/graphcheck.py:223` 中：

```python
inputs_embeds = torch.cat(
    [bos_embeds, claim_embedding, doc_embedding, inputs_embeds], dim=0
)
```

这意味着，一条训练样本最终输入给 LLM 的序列，在概念上是：

```text
[BOS, CLAIM_GRAPH, DOC_GRAPH, 文本prompt, [/INST], 标签token, </s>]
```

所以投影头的输出并不是喂给分类头，也不是和 logits 直接做相似度，而是被当成：

- 第一个图条件位置：claim 图向量；
- 第二个图条件位置：doc 图向量。

它们和普通 token embedding 的地位非常接近：

- 都出现在 `inputs_embeds` 序列里；
- 都会进入 LLM 的自注意力层；
- 都会影响后续标签 token 的生成。

这其实就是当前项目把图信息接入 LLM 的核心机制。

---

## 10. GNN 和投影头在训练中分别学什么

训练循环见 `train.py:81` 到 `train.py:132`。

### 10.1 哪些参数会被更新

LLM 初始化后立刻被冻结：

```python
for name, param in model.named_parameters():
    param.requires_grad = False
```

见 `model/graphcheck.py:59` 到 `model/graphcheck.py:61`。

优化器只收集 `requires_grad=True` 的参数：

```python
params = [p for _, p in model.named_parameters() if p.requires_grad]
```

见 `train.py:82`。

因此，当前项目真正训练的重点是：

- `graph_encoder`
- `projector`

而不是 LLM 本体。

### 10.2 loss 是怎样把信号传回 GNN 和投影头的

训练时模型最终返回的是：

```python
outputs.loss
```

见 `model/graphcheck.py:269` 到 `model/graphcheck.py:281`。

这个 loss 只监督末尾的标签 token，即 `support` / `unsupport` 和 EOS，而不监督前面的 prompt 位置。

于是梯度链路可以概念化成：

```text
标签预测误差
    -> 冻结 LLM 内部反向传播到输入 embedding 侧
    -> claim/doc 两个图向量位置收到梯度
    -> projector 收到梯度
    -> GNN 收到梯度
    -> 更新 graph_encoder 和 projector 参数
```

因为 LLM 权重被冻结，误差信号不会用于改写 LLM 主体，而是主要推动图模块学会：

- 图里哪些信息对标签判断重要；
- 怎样把这些信息整理成有判别力的图表示；
- 怎样把图表示变成冻结 LLM 最容易利用的 embedding 形式。

### 10.3 两者分工非常明确

可以把训练目标拆成两个子问题：

#### GNN 在学什么

GNN 在学：

- 如何利用节点语义、边语义和图结构；
- 如何把 claim 图编码成与真假判断有关的图摘要；
- 如何把 document 图编码成与证据支持关系有关的图摘要。

#### projector 在学什么

projector 在学：

- 如何把这两个图摘要重新表达成 LLM embedding 空间中的“可读信号”；
- 如何让冻结 LLM 在看到这两个额外向量后，更容易生成正确标签。

所以如果你把 GNN 理解成“图信息提取器”，那么 projector 就可以理解成“图信息翻译器”。

---

## 11. 为什么说 projector 是整个方案里很关键的桥

如果只有 GNN 没有 projector，会出现两个问题：

1. 维度不匹配，无法直接拼接；
2. 语义空间不匹配，即使能拼接，冻结 LLM 也未必能理解这些图向量。

如果只有 projector 没有 GNN，也不行，因为：

- 原始图不是单个固定向量；
- 需要先通过图消息传递把节点、边和结构上下文整合起来；
- 否则 projector 没有高质量图摘要可投影。

因此，这两部分不是替代关系，而是串联关系：

```text
GNN 负责提取图语义
projector 负责把图语义翻译给 LLM
```

---

## 12. 结合当前实现，可以把这套机制理解成什么

从工程角度看，GraphCheck 更像是在做一种轻量的图条件适配：

- 不微调整个大模型；
- 只训练图模块和适配头；
- 用很小的额外序列开销把图知识塞进 LLM。

从表示学习角度看，它做的是：

1. 用文本编码器给节点和边初始化语义表示；
2. 用 GNN 把局部结构和关系传播到节点表示中；
3. 用 mean pooling 压成图级摘要；
4. 用投影头把图摘要映射到 LLM 词向量空间；
5. 让冻结 LLM 使用这两个图摘要去辅助生成最终标签。

从任务角度看，它最终服务的是一个生成式二分类：

- 输入：claim + document + 两个图摘要；
- 输出：`support` 或 `unsupport`。

---

## 13. 用最简洁的话分别总结 GNN 和投影头

### GNN 的一句话定义

> GNN 把节点文本、关系文本和图结构融合起来，生成每张 claim/doc 图的图级摘要向量。

### 投影头的一句话定义

> 投影头把图级摘要向量变成冻结 LLM 的 embedding 空间中的两个条件向量，使图信息能够作为额外上下文注入语言模型。

---

## 14. 你读源码时最该盯住的几个位置

如果你想把上面的理解和代码逐行对上，建议重点看这几个文件：

- `graph_build.py:158`：节点文本和边文本如何编码成图特征
- `dataset/utils/modeling.py:33`：文本嵌入模型怎么做 mean pooling 和归一化
- `model/gnn.py:6`：`GraphTransformer` 结构
- `model/gnn.py:69`：`GAT` 结构
- `model/graphcheck.py:104`：GNN 输出如何做图级池化
- `model/graphcheck.py:83`：投影头结构
- `model/graphcheck.py:167`：图向量如何经过投影头
- `model/graphcheck.py:220`：投影后的图向量如何拼入 LLM 输入
- `train.py:82`：为什么只有图模块和投影头会被优化器更新

---

## 15. 最后给出一个总流程图

```text
claim/doc 三元组
    -> textualize_graph()
    -> 节点文本 + 边文本
    -> sentence transformer 编码
    -> PyG Data(x, edge_index, edge_attr)
    -> GNN(graph_encoder)
    -> 节点级表示
    -> mean pooling
    -> claim 图向量 / doc 图向量
    -> projector
    -> LLM embedding 维度中的两个向量
    -> 拼接到 [BOS] 后、文本前
    -> 冻结 LLM
    -> 生成 support / unsupport
    -> loss.backward()
    -> 更新 GNN + projector
```

如果你只记住这张图，大方向就不会错。

---

## 16. 图在 GraphCheck 里的作用到底是什么

如果只用一句话概括图的作用，可以写成：

> 图不是替代文本，而是给冻结 LLM 提供一份结构化补充证据，帮助它更准确地判断 document 是否 support claim。

文本输入更擅长表达顺序化的自然语言内容，例如：

- 句子里提到了哪些实体；
- 某段话是怎样写出来的；
- 某些词在上下文中如何共现。

而图输入更擅长表达结构化关系，例如：

- 谁和谁之间存在关系；
- 关系的类型是什么；
- 这些关系是否在 claim 和 document 之间一致、冲突或缺失。

因此在当前项目里：

- `Claim + Document` 提供自然语言证据；
- `claim_kg + doc_kg` 提供结构化证据；
- GraphCheck 的目标是把这两种证据联合起来，最终生成 `support` 或 `unsupport`。

从实现上看，这份结构化证据并不是以“离散图节点列表”的方式送进 LLM，而是先压缩成两个图向量，再作为两个额外条件位置拼到 LLM 输入序列中，见 `model/graphcheck.py:220`。

---

## 17. `claim_kg` 和 `doc_kg` 是怎么从三元组一步步变成向量图的

这一部分是理解图作用的基础。因为模型真正处理的并不是原始三元组字符串，而是已经向量化后的图对象。

### 17.1 原始输入：每条样本有两张知识图

在 `dataset/utils/dataset.py:54` 到 `dataset/utils/dataset.py:59` 中，每条样本会同时取出：

- `claim_kg`
- `doc_kg`

其中：

- `claim_kg` 表示从 claim 中抽取出的知识图；
- `doc_kg` 表示从 document 中抽取出的知识图。

它们的原始形式可以理解为若干条三元组：

```text
(head, relation, tail)
```

例如：

```text
(paris, capital_of, france)
```

### 17.2 第一步：三元组被拆成节点表和边表

在 `graph_build.py:35` 的 `textualize_graph()` 中，代码会把原始图拆成两张表：

- 节点表 `nodes`
- 边表 `edges`

节点表里主要有：

- `node_id`
- `node_attr`

边表里主要有：

- `src`
- `edge_attr`
- `dst`

也就是说：

- 实体文本变成节点属性；
- 关系文本变成边属性；
- 图的连接关系通过 `src -> dst` 保留下来。

### 17.3 第二步：节点文本和边文本分别编码成向量

在 `graph_build.py:159` 到 `graph_build.py:161` 中：

```python
x = text2embedding(model, tokenizer, device, nodes.node_attr.tolist())
e = text2embedding(model, tokenizer, device, edges.edge_attr.tolist())
```

这里的含义是：

- 所有节点文本被编码成节点向量矩阵 `x`
- 所有边关系文本被编码成边向量矩阵 `e`

编码器定义在 `dataset/utils/modeling.py:33`，它使用的是 `sentence-transformers/all-roberta-large-v1`：

1. 先把文本送入 `AutoModel`
2. 再做 mean pooling
3. 最后做 L2 归一化

所以在进入 GNN 之前：

- 每个节点已经有一个语义向量；
- 每条边也已经有一个关系语义向量。

### 17.4 第三步：组装成 PyG 图对象

在 `graph_build.py:165` 和 `graph_build.py:184` 中，代码构造：

```python
Data(x=x, edge_index=edge_index, edge_attr=e, num_nodes=len(nodes))
```

这一步以后，图就不再只是文本列表，而变成了标准的图神经网络输入：

- `x`：节点特征矩阵
- `edge_index`：边连接关系
- `edge_attr`：边特征矩阵

### 17.5 第四步：训练时读入的已经是“向量图”

在 `dataset/utils/dataset.py:64` 到 `dataset/utils/dataset.py:69` 中，训练阶段直接读取 `.pt` 图文件。

因此 `GraphCheck.encode_graphs()` 拿到的并不是生的三元组，而是已经向量化后的 PyG 图对象，见 `model/graphcheck.py:105`。

可以把这条链简化记成：

```text
三元组
-> 节点文本 / 关系文本
-> sentence transformer 编码
-> 节点向量 + 边向量
-> PyG Data 图对象
-> GNN
```

---

## 18. 为什么“只有两个图向量”也能影响最终的 `support` / `unsupport`

这是当前实现里最容易让人误解的地方。

很多人会直觉觉得：

> 一整张图只压成两个向量，会不会信息太少，怎么还能影响最终判断？

答案是：会有信息压缩，但依然能起作用，因为当前任务最终只需要做一个很小的决策：

```text
document 是否支持 claim
```

### 18.1 GNN 并不是直接输出给 LLM，而是先做图摘要

在 `model/graphcheck.py:110` 到 `model/graphcheck.py:115` 中，GNN 先输出节点级表示。

随后在 `model/graphcheck.py:130` 到 `model/graphcheck.py:139` 中，代码做平均池化，把节点级表示压成：

- 一个 `claim` 图向量
- 一个 `doc` 图向量

也就是说，当前实现选择的是：

- 不把每个节点都变成单独 token；
- 不把整张图展开成长序列；
- 而是只保留“整张图的摘要”。

### 18.2 这两个图向量会被当成两个额外的软 token

在 `model/graphcheck.py:167` 到 `model/graphcheck.py:170` 中，这两个图摘要先经过 `projector`，被映射到 LLM 的 embedding 维度。

随后在 `model/graphcheck.py:220` 到 `model/graphcheck.py:223` 中，它们被拼进输入序列：

```text
[BOS] [claim图向量] [doc图向量] [文本prompt] ...
```

因此，从 LLM 视角看，这两个向量并不是普通词表里的 token，而是两个“连续提示 token”或“软 token”。

虽然它们不对应词表中的真实单词，但它们会像普通 token embedding 一样：

- 进入自注意力计算；
- 影响后续位置的 hidden states；
- 影响最终标签位置的 logits。

### 18.3 任务本身只需要一个高层决策，而不是复原整张图

这里的关键不是让 LLM 完整重建图，而是让它回答一个二分类问题：

- support
- unsupport

对这种任务来说，真正重要的往往不是全部局部细节，而是一个较高层的结构结论，例如：

- claim 图中关键主体-关系-客体是什么；
- doc 图中关键主体-关系-客体是什么；
- 两边结构是否一致、冲突或缺失。

如果这两个图向量已经浓缩了这些高层差异，那么它们就足以改变最终答案位置的表示。

### 18.4 这个设计的优点和代价

优点：

- 序列长度开销很小；
- 不会把大图展开成超长上下文；
- 易于和冻结 LLM 拼接；
- 训练更轻量。

代价：

- 图的细粒度局部信息会被压缩；
- 所有图知识都必须塞进两个摘要向量；
- 存在明显的信息瓶颈。

所以更准确的说法不是“两个图向量等于整张图”，而是：

> 两个图向量提供的是对最终二分类最有用的图摘要信号。

---

## 19. 大模型是如何识别或者说使用图向量的

大模型并不会把图向量“识别成某个词”，而是把它们当作序列中的两个额外 embedding 位置来处理。

### 19.1 图向量不是通过 tokenizer 进入模型的

普通 token 的路径是：

```text
tokenizer -> token_id -> embedding
```

图向量的路径是：

```text
GNN -> projector -> embedding
```

也就是说，图向量绕过了词表，直接进入了 LLM 的 embedding 空间。

### 19.2 图向量被放在输入序列前面

在 `model/graphcheck.py:220` 中，最终顺序是：

```text
[BOS] [claim图向量] [doc图向量] [文本prompt] [标签token...]
```

这意味着：

- 文本位置能看到图位置；
- 标签位置也能看到图位置；
- 当模型预测 `support` 或 `unsupport` 时，可以对这两个图位置做注意力读取。

### 19.3 LLM 的自注意力会把这两个向量当作条件前缀

冻结 LLM 在前向传播时，不会区分“这是词表 token”还是“这是投影后的图向量”。

对它来说，只要某个位置给出了合法的 embedding，它就会：

- 为该位置计算 hidden state；
- 在自注意力中把该位置作为可读取的上下文；
- 让后续位置根据这个上下文更新自己的表示。

因此，当模型在答案位置预测标签时，它本质上是在综合：

- 文本 prompt 提供的语言信息；
- claim 图向量提供的 claim 结构摘要；
- doc 图向量提供的 document 结构摘要。

### 19.4 `projector` 的核心作用，是让冻结 LLM“看得懂”图向量

GNN 输出的向量在图表示空间里，LLM 的输入 embedding 在语言模型空间里。

就算两个向量维度相同，它们也不天然在同一个语义空间里。

所以 `projector` 真正做的不是简单的“改维度”，而是：

> 把图摘要翻译成冻结 LLM 可以利用的连续提示向量。

这就是为什么即便 LLM 本体不更新，图向量依然能发挥作用：

- LLM 负责读取这些向量；
- `GNN + projector` 负责学会产出对 LLM 有意义的向量。

可以把它类比成 soft prompt 或 prefix tuning：

- 不是修改大模型参数；
- 而是学习“怎样构造前缀向量去引导大模型”。

---

## 20. 用一个具体样本说明图到底帮了模型什么

下面用两个简化样例来说明图的实际作用。

### 20.1 反例：图帮助发现结构冲突

假设输入是：

```text
Claim: 巴黎是德国首都。
Document: 柏林是德国首都。巴黎是法国首都。
```

claim 图可能抽成：

```text
(巴黎, 是首都, 德国)
```

doc 图可能抽成：

```text
(柏林, 是首都, 德国)
(巴黎, 是首都, 法国)
```

如果只看词面，模型会同时看到：

- 巴黎
- 德国
- 首都

这些关键词高度重叠，存在被表面共现误导的风险。

但图把“谁和谁通过什么关系相连”明确固定下来以后，结构差异就变得更清楚：

- claim 结构是：巴黎 -> 首都 -> 德国
- doc 结构是：巴黎 -> 首都 -> 法国，以及 柏林 -> 首都 -> 德国

这样 GNN 压缩出的两个图摘要就更容易体现：

- claim 的核心关系
- doc 的核心关系
- 两者在关键实体关系上是冲突的

于是图会推动模型更倾向于生成：

```text
unsupport
```

### 20.2 正例：图帮助建立支持链条

再看一个支持样例：

```text
Claim: 爱因斯坦出生于德国。
Document: 爱因斯坦出生在德国乌尔姆。
```

claim 图可能是：

```text
(爱因斯坦, 出生于, 德国)
```

doc 图可能是：

```text
(爱因斯坦, 出生于, 乌尔姆)
(乌尔姆, 位于, 德国)
```

这里 document 不一定直接给出和 claim 完全同形的句子，但图结构可以保留一条支持链：

```text
爱因斯坦 -> 出生于 -> 乌尔姆 -> 位于 -> 德国
```

GNN 的消息传递有机会把这种局部关系链压进 doc 图摘要中，使它对“支持 claim”这一决策更有帮助。

换句话说，图不只是抽关键词，而是在尽可能保留：

- 事实关系的连接方式；
- 关系之间是否能组成支持路径；
- claim 与 document 的结构是否一致。

---

## 21. 一句话总结这三部分

把这三部分合起来，你可以这样记：

- `claim_kg` 和 `doc_kg` 先从三元组变成带节点向量和边向量的 PyG 图对象；
- GNN 把每张图压成一个图摘要，投影头再把它翻译到 LLM embedding 空间；
- 冻结 LLM 把这两个图摘要当作两个条件前缀位置，在生成 `support` / `unsupport` 时把它们作为结构化证据使用。

如果要再压缩成一句最核心的话：

> GraphCheck 不是让 LLM 直接“读图文件”，而是先把图压成两个它能读懂的连续提示向量，再利用这些向量辅助完成事实支持判断。

---

## 22. GNN 在输入和输出之间到底做了什么

在本项目里，GNN 的直接任务不是输出分类结果，也不是直接输出 `support` / `unsupport`，而是：

> 根据图结构和边关系，更新每个节点的表示向量。

也就是说，GNN 做的是“节点表示学习”。

### 22.1 GNN 的三个输入分别是什么

在 `model/graphcheck.py:110` 到 `model/graphcheck.py:115` 中，图编码器的调用方式是：

```python
claim_n_embeds, _ = self.graph_encoder(
    claim_kg.x, claim_kg.edge_index.long(), claim_kg.edge_attr
)
```

这说明 GNN 接收三个核心输入：

- `x`：节点特征矩阵
- `edge_index`：边连接关系
- `edge_attr`：边特征矩阵

如果按默认配置 `gnn_in_dim=1024` 来理解，那么：

- `x` 的形状通常是 `[num_nodes, 1024]`
- `edge_index` 的形状通常是 `[2, num_edges]`
- `edge_attr` 的形状通常是 `[num_edges, 1024]`

### 22.2 在前向传播过程中，GNN 真正会改什么

这一点非常关键。

在当前实现里，GNN 真正持续被更新的是：

- `x`，也就是节点表示

而下面两项主要是作为条件参与计算：

- `edge_index`：告诉模型图里谁和谁相连
- `edge_attr`：告诉模型这些连接对应什么关系语义

所以更准确地说，GNN 做的不是“修改图结构”，而是：

> 利用图结构 `edge_index` 和边关系 `edge_attr`，去更新每个节点的表示 `x`。

### 22.3 `GraphTransformer` 在输入与输出之间做了什么

在 `model/gnn.py:58` 到 `model/gnn.py:66` 中，`GraphTransformer.forward()` 的主干是：

```python
for i, conv in enumerate(self.convs[:-1]):
    x = conv(x, edge_index=adj_t, edge_attr=edge_attr)
    x = self.bns[i](x)
    x = F.relu(x)
    x = F.dropout(x, p=self.dropout, training=self.training)
x = self.convs[-1](x, edge_index=adj_t, edge_attr=edge_attr)
return x, edge_attr
```

这表示每一层都在做：

```text
TransformerConv -> BatchNorm -> ReLU -> Dropout
```

最后一层只输出最终节点表示，不再接激活和 dropout。

这里真正被逐层更新的是 `x`，也就是每个节点的向量表示。

### 22.4 `GAT` 在输入与输出之间做了什么

在 `model/gnn.py:104` 到 `model/gnn.py:126` 中，`GAT.forward()` 的主干是：

```python
for i, conv in enumerate(self.convs[:-1]):
    x, attn_weights = conv(...)
    x = self.bns[i](x)
    x = F.relu(x)
    x = F.dropout(x, p=self.dropout, training=self.training)

x, attn_weights = self.convs[-1](...)
return x, edge_attr
```

它和 `GraphTransformer` 的整体节奏类似，只是核心卷积层换成了 `GATConv`。

也就是说，GAT 也是：

- 使用图结构和边特征更新节点表示 `x`
- 额外保留每层注意力权重，便于分析模型关注了哪些边

但在输出层面，真正重要的仍然是更新后的节点表示矩阵 `x`。

---

## 23. GNN 输出的格式是什么

这一点也很重要，因为很多人第一次看会误以为 GNN 一出来就是“图向量”。

实际上不是。

GNN 的直接输出是：

> 每个节点的最终表示矩阵

### 23.1 输出张量 `x` 的形状

不管是 `GraphTransformer` 还是 `GAT`，它们最终都返回：

```python
return x, edge_attr
```

见 `model/gnn.py:66` 和 `model/gnn.py:126`。

其中真正关键的是第一个返回值 `x`，它的形状通常是：

```text
[num_nodes, out_channels]
```

如果当前配置里 `out_channels = gnn_hidden_dim = 1024`，那它就可以写成：

```text
[num_nodes, 1024]
```

举个例子，如果一张图有 5 个节点，那么输出可能是：

```text
x.shape = [5, 1024]
```

这表示：

- 第 1 行是第 1 个节点的最终表示
- 第 2 行是第 2 个节点的最终表示
- ...
- 第 5 行是第 5 个节点的最终表示

所以，GNN 的直接输出不是“整张图一个向量”，而是：

> 每个节点一个向量

### 23.2 第二个返回值 `edge_attr`

当前两个 GNN 类都把 `edge_attr` 原样返回了：

```python
return x, edge_attr
```

这说明：

- `edge_attr` 在前向中参与了消息传递
- 但在当前实现里，它本身并没有被更新成新的边表示再返回
- `GraphCheck.encode_graphs()` 也没有继续使用第二个返回值

因此，在本项目里更应该把 GNN 的核心输出理解为：

```text
节点表示矩阵 x
```

---

## 24. 对输出的 `x` 做池化时发生了什么

由于 GraphCheck 最终只想给 LLM 两个图摘要，而不是一整串节点向量，所以 GNN 输出后还要做一步图级池化。

这一步发生在 `model/graphcheck.py:130` 和 `model/graphcheck.py:137`：

```python
claim_embeds = scatter(claim_n_embeds, claim_kg.batch, dim=0, reduce="mean")
doc_embeds = scatter(doc_n_embeds, doc_kg.batch, dim=0, reduce="mean")
```

这里使用的是：

```text
mean pooling
```

也就是按图编号分组，对同一张图里的所有节点向量取平均。

### 24.1 池化前的形状

先看 `claim_n_embeds`，它是 GNN 输出的节点表示矩阵：

```text
claim_n_embeds.shape = [总节点数, gnn_hidden_dim]
```

如果一个 batch 中有多张图，PyG 会先把它们拼在一起，所以这里的“总节点数”是 batch 中所有图节点数之和。

例如：

- 第 1 张图 4 个节点
- 第 2 张图 5 个节点
- 第 3 张图 3 个节点

那么：

```text
claim_n_embeds.shape = [12, 1024]
```

### 24.2 `claim_kg.batch` 的作用

`claim_kg.batch` 会告诉你每个节点属于哪张图。

例如：

```text
[0,0,0,0, 1,1,1,1,1, 2,2,2]
```

它表示：

- 前 4 个节点属于第 0 张图
- 中间 5 个节点属于第 1 张图
- 最后 3 个节点属于第 2 张图

`scatter(..., reduce="mean")` 就是根据这个图编号做分组平均。

### 24.3 池化具体做了什么

以某张图 `g` 为例，池化后的图向量可以理解为：

```text
graph_embed[g] = 这张图所有节点向量的平均
```

也就是说：

- 不保留每个节点单独的表示；
- 把整张图压成 1 个摘要向量；
- 这个摘要向量可以看作“整张图的平均语义表示”。

### 24.4 池化后的形状

池化后：

```text
claim_embeds.shape = [图数, gnn_hidden_dim]
doc_embeds.shape   = [图数, gnn_hidden_dim]
```

由于每条样本通常对应 1 张 claim 图和 1 张 doc 图，所以在训练时通常可以直接理解为：

```text
claim_embeds.shape = [batch_size, gnn_hidden_dim]
doc_embeds.shape   = [batch_size, gnn_hidden_dim]
```

按默认配置 `gnn_hidden_dim=1024` 来看：

- 如果 `batch_size=8`

那么通常就是：

```text
claim_embeds.shape = [8, 1024]
doc_embeds.shape   = [8, 1024]
```

### 24.5 单图情况下的退化写法

如果当前不是 batched 图，而只是单张图，代码会退化成：

```python
claim_embeds = claim_n_embeds.mean(dim=0, keepdim=True)
doc_embeds = doc_n_embeds.mean(dim=0, keepdim=True)
```

这时：

- 池化前形状：`[num_nodes, gnn_hidden_dim]`
- 池化后形状：`[1, gnn_hidden_dim]`

### 24.6 一个最小例子

假设一张图有 3 个节点，GNN 输出：

```text
x =
[
  [1, 2],
  [3, 4],
  [5, 6]
]
```

mean pooling 后：

```text
graph_embed = [(1+3+5)/3, (2+4+6)/3]
           = [3, 4]
```

形状从：

```text
[3, 2]
```

变成：

```text
[1, 2]
```

也就是从“3 个节点、每个节点 2 维”，压成了“1 张图、1 个 2 维摘要向量”。

---

## 25. 池化后的结果为什么还要再过投影头

池化后得到的是：

```text
[batch_size, gnn_hidden_dim]
```

但 LLM 需要的输入 embedding 维度是：

```text
llm_embed_dim
```

因此后面还会执行：

```python
claim_embeds = self.projector(claim_embeds)
doc_embeds = self.projector(doc_embeds)
```

见 `model/graphcheck.py:167` 到 `model/graphcheck.py:170`。

投影后形状会变成：

```text
[batch_size, llm_embed_dim]
```

这时它们才真正能被作为两个图条件位置拼入 `inputs_embeds`。

---

## 26. 一句话总结这一段

把这一部分压缩成最关键的三句话，就是：

- GNN 真正更新的是节点表示 `x`，不是图结构本身；
- GNN 的直接输出格式是 `[num_nodes, out_channels]`，也就是“每个节点一个向量”；
- GraphCheck 再对这些节点向量做 mean pooling，把它压成 `[batch_size, gnn_hidden_dim]` 的图级表示，最后再送入投影头和 LLM。
