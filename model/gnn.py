import torch  # 导入 PyTorch 主库，后面定义模型类和张量操作都要用到。
import torch.nn.functional as F  # 导入常用函数式接口，这里主要用 ReLU 和 dropout。
from torch_geometric.nn import TransformerConv, GATConv  # 导入两种图卷积层：图 Transformer 和图注意力网络。


class GraphTransformer(torch.nn.Module):  # 定义基于 TransformerConv 的图编码器。
    def __init__(  # 初始化图 Transformer 的网络结构。
        self,  # 当前实例自身。
        in_channels,  # 输入节点特征维度。
        hidden_channels,  # 中间隐藏层维度。
        out_channels,  # 最终输出节点表示维度。
        num_layers,  # 图卷积层数。
        dropout,  # dropout 比例。
        num_heads=-1,  # 多头注意力的头数。
    ):
        super(GraphTransformer, self).__init__()  # 初始化父类 torch.nn.Module。
        self.convs = torch.nn.ModuleList()  # 创建一个可注册参数的列表，用来存放所有图卷积层。
        # TransformerConv 同时利用节点特征和边特征做消息传递。
        self.convs.append(  # 先加入第一层图 Transformer 卷积。
            TransformerConv(
                in_channels=in_channels,  # 第一层输入维度来自原始节点特征维度。
                out_channels=hidden_channels // num_heads,  # 每个头输出 hidden_channels/heads，合起来得到 hidden_channels。
                heads=num_heads,  # 指定多头注意力头数。
                edge_dim=in_channels,  # 指定边特征维度，这里与输入节点维度一致。
                dropout=dropout,  # 给注意力内部使用同样的 dropout 比例。
            )
        )
        self.bns = torch.nn.ModuleList()  # 创建 BatchNorm 层列表，与中间卷积层一一对应。
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))  # 为第一层输出添加一层一维批归一化。
        for _ in range(num_layers - 2):  # 构造中间若干层隐藏层卷积。
            self.convs.append(  # 为每个中间层加入一个 TransformerConv。
                TransformerConv(
                    in_channels=hidden_channels,  # 中间层输入维度来自上一层 hidden 输出。
                    out_channels=hidden_channels // num_heads,  # 每个头输出 hidden_channels/heads，总输出仍为 hidden_channels。
                    heads=num_heads,  # 中间层也使用同样的多头数。
                    edge_dim=in_channels,  # 边特征维度仍然固定为原始输入边向量维度。
                    dropout=dropout,  # 中间层继续使用同样的 dropout 比例。
                )
            )
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))  # 为这个中间层再配一个 BN。
        self.convs.append(  # 最后再加入输出层卷积。
            TransformerConv(
                in_channels=hidden_channels,  # 输出层输入来自最后一个隐藏层。
                out_channels=out_channels // num_heads,  # 每个头输出 out_channels/heads，总输出为 out_channels。
                heads=num_heads,  # 输出层继续使用同样的头数。
                edge_dim=in_channels,  # 输出层同样显式使用边特征。
                dropout=dropout,  # 输出层也保留 dropout 配置。
            )
        )
        self.dropout = dropout  # 保存 dropout 比例，供 forward 中直接调用。

    def reset_parameters(self):  # 提供一个统一重置参数的方法。
        for conv in self.convs:  # 逐层遍历所有图卷积层。
            conv.reset_parameters()  # 调用每个卷积层自己的参数重置逻辑。
        for bn in self.bns:  # 再遍历所有 BN 层。
            bn.reset_parameters()  # 重置 BN 的缩放和偏置参数。

    def forward(self, x, adj_t, edge_attr):  # 前向传播：输入节点特征、边索引和边特征。
        # 前几层都做 conv -> BN -> ReLU -> dropout，最后一层只输出表示不过激活。
        for i, conv in enumerate(self.convs[:-1]):  # 先遍历除最后一层外的所有卷积层。
            x = conv(x, edge_index=adj_t, edge_attr=edge_attr)  # 用当前层结合图结构和边特征更新节点表示。
            x = self.bns[i](x)  # 对当前层输出做批归一化，稳定训练。
            x = F.relu(x)  # 对归一化后的节点表示做 ReLU 激活。
            x = F.dropout(x, p=self.dropout, training=self.training)  # 仅在训练时对节点表示做 dropout。
        x = self.convs[-1](x, edge_index=adj_t, edge_attr=edge_attr)  # 最后一层只输出最终节点表示，不再接 BN/ReLU/dropout。
        return x, edge_attr  # 返回节点表示，以及原样返回边特征以保持接口统一。


class GAT(torch.nn.Module):  # 定义基于 GATConv 的图注意力编码器。
    def __init__(  # 初始化 GAT 网络结构。
        self,  # 当前实例自身。
        in_channels,  # 输入节点特征维度。
        hidden_channels,  # 隐藏层输出维度。
        out_channels,  # 最终输出节点表示维度。
        num_layers,  # GAT 层数。
        dropout,  # dropout 比例。
        num_heads=4,  # 默认使用 4 个注意力头。
    ):
        super(GAT, self).__init__()  # 初始化父类 torch.nn.Module。
        self.convs = torch.nn.ModuleList()  # 创建可注册参数的列表，用来存放所有 GAT 层。
        # concat=False 表示多头注意力最后不拼接，而是保持输出维度稳定。
        self.convs.append(  # 先加入第一层 GATConv。
            GATConv(in_channels, hidden_channels, heads=num_heads, concat=False)  # 第一层把输入维度映射到 hidden_channels。
        )
        self.bns = torch.nn.ModuleList()  # 创建 BN 层列表，与中间卷积层输出对应。
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))  # 为第一层输出添加 BN。
        for _ in range(num_layers - 2):  # 构造中间若干层隐藏层 GAT。
            self.convs.append(  # 为每个中间层再加入一个 GATConv。
                GATConv(hidden_channels, hidden_channels, heads=num_heads, concat=False)  # 保持隐藏维度不变。
            )
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))  # 给每个中间层输出再加一个 BN。
        self.convs.append(  # 最后加入输出层 GATConv。
            GATConv(hidden_channels, out_channels, heads=num_heads, concat=False)  # 输出层把 hidden_channels 映射到 out_channels。
        )
        self.dropout = dropout  # 保存 dropout 比例，供 forward 中调用。
        self.attn_weights = None  # 预留一个成员变量，用来缓存最后一层注意力权重。

    def reset_parameters(self):  # 提供统一的参数重置接口。
        for conv in self.convs:  # 逐层遍历所有 GATConv。
            conv.reset_parameters()  # 重置当前卷积层的参数。
        for bn in self.bns:  # 再遍历全部 BN 层。
            bn.reset_parameters()  # 重置 BN 参数。

    def forward(self, x, edge_index, edge_attr):  # 前向传播：输入节点特征、边连接和边特征。
        attn_weights_list = []  # 用列表缓存每一层返回的注意力权重，便于后续分析。
        # 这里保留每层注意力权重，便于后续分析或可视化，但主训练流程只使用最终节点表示。
        for i, conv in enumerate(self.convs[:-1]):  # 先遍历除最后一层外的所有 GAT 层。
            x, attn_weights = conv(  # 当前层返回更新后的节点表示和对应的注意力权重。
                x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                return_attention_weights=True,
            )
            attn_weights_list.append(attn_weights[1])  # 只保存注意力权重张量本身，不保存边索引那一部分。
            x = self.bns[i](x)  # 对当前层输出做 BN，帮助训练更稳定。
            x = F.relu(x)  # 再做 ReLU 激活。
            x = F.dropout(x, p=self.dropout, training=self.training)  # 训练时对节点表示做 dropout。

        x, attn_weights = self.convs[-1](  # 最后一层同样返回节点表示和注意力权重。
            x, edge_index=edge_index, edge_attr=edge_attr, return_attention_weights=True
        )
        attn_weights_list.append(attn_weights[1])  # 把最后一层的注意力权重也加入列表。
        # 当前类只把最后一层的注意力权重挂到实例上，方便外部读取。
        self.attn_weights = attn_weights_list[-1]  # 缓存最后一层的注意力权重，供解释性分析使用。

        return x, edge_attr  # 返回最终节点表示，并原样返回边特征以保持接口一致。


# 训练脚本通过这个映射按名字选择具体的图编码器实现。
load_gnn_model = {  # 建立字符串名称到具体 GNN 类的映射表。
    "gat": GAT,  # 选择 "gat" 时，实例化 GAT 编码器。
    "gt": GraphTransformer,  # 选择 "gt" 时，实例化 GraphTransformer 编码器。
}
