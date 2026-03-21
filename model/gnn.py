import torch
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, GATConv


class GraphTransformer(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        num_layers,
        dropout,
        num_heads=-1,
    ):
        super(GraphTransformer, self).__init__()
        self.convs = torch.nn.ModuleList()
        # TransformerConv 同时利用节点特征和边特征做消息传递。
        self.convs.append(
            TransformerConv(
                in_channels=in_channels,
                out_channels=hidden_channels // num_heads,
                heads=num_heads,
                edge_dim=in_channels,
                dropout=dropout,
            )
        )
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(
                TransformerConv(
                    in_channels=hidden_channels,
                    out_channels=hidden_channels // num_heads,
                    heads=num_heads,
                    edge_dim=in_channels,
                    dropout=dropout,
                )
            )
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(
            TransformerConv(
                in_channels=hidden_channels,
                out_channels=out_channels // num_heads,
                heads=num_heads,
                edge_dim=in_channels,
                dropout=dropout,
            )
        )
        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t, edge_attr):
        # 前几层都做 conv -> BN -> ReLU -> dropout，最后一层只输出表示不过激活。
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index=adj_t, edge_attr=edge_attr)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index=adj_t, edge_attr=edge_attr)
        return x, edge_attr


class GAT(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        num_layers,
        dropout,
        num_heads=4,
    ):
        super(GAT, self).__init__()
        self.convs = torch.nn.ModuleList()
        # concat=False 表示多头注意力最后不拼接，而是保持输出维度稳定。
        self.convs.append(
            GATConv(in_channels, hidden_channels, heads=num_heads, concat=False)
        )
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(hidden_channels, hidden_channels, heads=num_heads, concat=False)
            )
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(
            GATConv(hidden_channels, out_channels, heads=num_heads, concat=False)
        )
        self.dropout = dropout
        self.attn_weights = None

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, edge_index, edge_attr):
        attn_weights_list = []
        # 这里保留每层注意力权重，便于后续分析或可视化，但主训练流程只使用最终节点表示。
        for i, conv in enumerate(self.convs[:-1]):
            x, attn_weights = conv(
                x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                return_attention_weights=True,
            )
            attn_weights_list.append(attn_weights[1])
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x, attn_weights = self.convs[-1](
            x, edge_index=edge_index, edge_attr=edge_attr, return_attention_weights=True
        )
        attn_weights_list.append(attn_weights[1])
        # 当前类只把最后一层的注意力权重挂到实例上，方便外部读取。
        self.attn_weights = attn_weights_list[-1]

        return x, edge_attr


# 训练脚本通过这个映射按名字选择具体的图编码器实现。
load_gnn_model = {
    "gat": GAT,
    "gt": GraphTransformer,
}
