import os
import sys
import torch
import pandas as pd
from dataset.utils.modeling import load_model, load_text2embedding
from tqdm import tqdm
from torch_geometric.data.data import Data
import os
import numpy as np
from sklearn.model_selection import train_test_split
import argparse

pythonpath = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
print(pythonpath)
sys.path.insert(0, pythonpath)


def parse_args():
    parser = argparse.ArgumentParser(description="GraphCheck")

    parser.add_argument(
        "--data_name", type=str, required=True, help="Name of the dataset folder"
    )
    parser.add_argument(
        "--project_root",
        type=str,
        default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
        help="Root directory of the project",
    )

    args = parser.parse_args()
    return args


def textualize_graph(graph):
    """
    把三元组变成节点表和边表
    """
    # 输入 graph 是一个三元组列表，每个元素通常形如 (head, relation, tail)。
    if not graph:
        # 如果图为空，则创建一个空节点表。
        nodes = pd.DataFrame(columns=["node_attr", "node_id"])
        # 如果图为空，则创建一个空边表。
        edges = pd.DataFrame(columns=["src", "edge_attr", "dst"])
        # 直接返回空节点表和空边表。
        return nodes, edges

    # 用字典给节点文本分配唯一编号。
    nodes = {}
    # 用列表暂存边，后面再统一转成 DataFrame。
    edges = []

    # 遍历图中的每一个三元组。
    for tri in graph:
        # 将三元组拆分为起点实体、关系文本和终点实体。
        src, edge_attr, dst = tri

        # 如果起点实体非空，就转小写并去掉首尾空格；否则用空格占位。
        src = src.lower().strip() if src else " "
        # 如果关系非空，就转小写并去掉首尾空格；否则用空格占位。
        edge_attr = edge_attr.lower().strip() if edge_attr else " "
        # 如果终点实体非空，就转小写并去掉首尾空格；否则用空格占位。
        dst = dst.lower().strip() if dst else " "

        # 如果起点实体还没有编号，就分配一个新的编号。
        if src not in nodes:
            nodes[src] = len(nodes)
        # 如果终点实体还没有编号，就分配一个新的编号。
        if dst not in nodes:
            nodes[dst] = len(nodes)

        # 将当前三元组对应的边加入边列表。
        edges.append(
            {
                # 起点节点使用分配好的整数编号。
                "src": nodes[src],
                # 边上保存的是关系文本。
                "edge_attr": edge_attr,
                # 终点节点同样使用整数编号。
                "dst": nodes[dst],
            }
        )

    # 将节点字典转换为 DataFrame，得到节点属性表。
    nodes = pd.DataFrame(nodes.items(), columns=["node_attr", "node_id"])
    # 将边列表转换为 DataFrame，得到边属性表。
    edges = pd.DataFrame(edges)

    # 返回标准化后的节点表和边表。
    return nodes, edges


def step_one():
    """
    为每条样本保存节点 CSV 和边 CSV
    :return:
    """
    # 为 claim 图的节点表创建保存目录。
    os.makedirs(f"{path}/{data_name}/nodes/claim", exist_ok=True)
    # 为 doc 图的节点表创建保存目录。
    os.makedirs(f"{path}/{data_name}/nodes/doc", exist_ok=True)
    # 为 claim 图的边表创建保存目录。
    os.makedirs(f"{path}/{data_name}/edges/claim", exist_ok=True)
    # 为 doc 图的边表创建保存目录。
    os.makedirs(f"{path}/{data_name}/edges/doc", exist_ok=True)

    # 逐条遍历数据集中的 claim_kg 和 doc_kg，并记录当前样本编号 i。
    for i, (claim_kg, doc_kg) in enumerate(
        tqdm(zip(claim_kgs, doc_kgs), total=len(dataset))
    ):
        # 将当前样本的 claim_kg 转成节点表和边表。
        claim_nodes, claim_edges = textualize_graph(claim_kg)
        # 把 claim 图的节点表保存成 CSV，文件名与样本编号一致。
        claim_nodes.to_csv(
            f"{path}/{data_name}/nodes/claim/{i}.csv",
            index=False,
            columns=["node_id", "node_attr"],
        )
        # 把 claim 图的边表保存成 CSV，包含起点、关系文本和终点。
        claim_edges.to_csv(
            f"{path}/{data_name}/edges/claim/{i}.csv",
            index=False,
            columns=["src", "edge_attr", "dst"],
        )

        # 将当前样本的 doc_kg 转成节点表和边表。
        doc_nodes, doc_edges = textualize_graph(doc_kg)
        # 把 doc 图的节点表保存成 CSV，文件名与样本编号一致。
        doc_nodes.to_csv(
            f"{path}/{data_name}/nodes/doc/{i}.csv",
            index=False,
            columns=["node_id", "node_attr"],
        )
        # 把 doc 图的边表保存成 CSV，包含起点、关系文本和终点。
        doc_edges.to_csv(
            f"{path}/{data_name}/edges/doc/{i}.csv",
            index=False,
            columns=["src", "edge_attr", "dst"],
        )


def step_two():
    """
    把节点/边文本编码成向量图
    :return:
    """
    def _encode_graph():
        # 提示当前开始执行图向量化阶段。
        print("Encoding graphs...")
        # 为编码后的 claim 图创建保存目录。
        os.makedirs(f"{path}/{data_name}/graphs/claim", exist_ok=True)
        # 逐个读取 claim 图的节点表和边表，并编码成 PyG 图对象。
        for i in tqdm(range(len(dataset))):
            # 读取第 i 个样本的 claim 节点表。
            nodes = pd.read_csv(f"{path}/{data_name}/nodes/claim/{i}.csv")
            # 读取第 i 个样本的 claim 边表。
            edges = pd.read_csv(f"{path}/{data_name}/edges/claim/{i}.csv")
            # 将节点文本编码为节点特征矩阵。
            x = text2embedding(model, tokenizer, device, nodes.node_attr.tolist())
            # 将边上的关系文本编码为边特征矩阵。
            e = text2embedding(model, tokenizer, device, edges.edge_attr.tolist())
            # 根据边表中的起点和终点构造图的连接关系。
            edge_index = torch.LongTensor([edges.src, edges.dst])
            # 组装成 PyTorch Geometric 的 Data 对象。
            data = Data(x=x, edge_index=edge_index, edge_attr=e, num_nodes=len(nodes))
            # 将编码后的 claim 图保存为 .pt 文件。
            torch.save(data, f"{path}/{data_name}/graphs/claim/{i}.pt")

        # 为编码后的 doc 图创建保存目录。
        os.makedirs(f"{path}/{data_name}/graphs/doc", exist_ok=True)
        # 逐个读取 doc 图的节点表和边表，并编码成 PyG 图对象。
        for i in tqdm(range(len(dataset))):
            # 读取第 i 个样本的 doc 节点表。
            nodes = pd.read_csv(f"{path}/{data_name}/nodes/doc/{i}.csv")
            # 读取第 i 个样本的 doc 边表。
            edges = pd.read_csv(f"{path}/{data_name}/edges/doc/{i}.csv")
            # 将节点文本编码为节点特征矩阵。
            x = text2embedding(model, tokenizer, device, nodes.node_attr.tolist())
            # 将边上的关系文本编码为边特征矩阵。
            e = text2embedding(model, tokenizer, device, edges.edge_attr.tolist())
            # 根据边表中的起点和终点构造图的连接关系。
            edge_index = torch.LongTensor([edges.src, edges.dst])
            # 组装成 PyTorch Geometric 的 Data 对象。
            data = Data(x=x, edge_index=edge_index, edge_attr=e, num_nodes=len(nodes))
            # 将编码后的 doc 图保存为 .pt 文件。
            torch.save(data, f"{path}/{data_name}/graphs/doc/{i}.pt")

    # 加载文本编码模型、分词器和运行设备。
    model, tokenizer, device = load_model()
    # 取出文本转向量函数，供后面的节点和边编码使用。
    text2embedding = load_text2embedding

    # 执行真正的图编码过程。
    _encode_graph()


def generate_split(num_nodes, path):
    """
    训练集、验证集、测试集划分
    :param num_nodes:
    :param path:
    :return:
    """
    # 生成从 0 到 num_nodes - 1 的全部样本索引。
    indices = np.arange(num_nodes)
    # 第一次切分：先拿出 60% 作为训练集，剩余 40% 暂存到 temp_data。
    train_indices, temp_data = train_test_split(indices, test_size=0.4, random_state=42)
    # 第二次切分：把剩余的 40% 平分成验证集和测试集。
    val_indices, test_indices = train_test_split(
        temp_data, test_size=0.5, random_state=42
    )
    # 打印训练集样本数量。
    print("# train samples: ", len(train_indices))
    # 打印验证集样本数量。
    print("# val samples: ", len(val_indices))
    # 打印测试集样本数量。
    print("# test samples: ", len(test_indices))

    # 为切分结果创建保存目录。
    os.makedirs(path, exist_ok=True)

    # 将训练集索引保存到单独的文本文件中。
    with open(f"{path}/train_indices.txt", "w") as file:
        file.write("\n".join(map(str, train_indices)))

    # 将验证集索引保存到单独的文本文件中。
    with open(f"{path}/val_indices.txt", "w") as file:
        file.write("\n".join(map(str, val_indices)))

    # 将测试集索引保存到单独的文本文件中。
    with open(f"{path}/test_indices.txt", "w") as file:
        file.write("\n".join(map(str, test_indices)))


if __name__ == "__main__":
    args = parse_args()

    project_root = args.project_root
    path = f"{project_root}/GraphCheck/dataset/extracted_KG"
    data_name = args.data_name

    dataset = pd.read_pickle(f"{path}/{data_name}/{data_name}.pkl")

    doc_kgs = dataset["doc_kg"]
    claim_kgs = dataset["claim_kg"]
    labels = dataset["label"]

    step_one()
    step_two()
    generate_split(len(dataset), f"{path}/{data_name}/split")
