import torch  # 核心张量库，用于文本特征编码与批处理。
from torch import nn  # 神经网络模块，这里也会用到 DataParallel。
import torch.nn.functional as F  # 函数式接口，这里主要用于向量归一化。
from transformers import (
    AutoModel,
    AutoTokenizer,
)  # Hugging Face 模型与分词器加载器。
from torch.utils.data import (
    DataLoader,
)  # 用于按 batch 处理分词后的文本，提高推理效率。
import numpy as np  # 预留的数值计算库，便于后续扩展使用。

pretrained_repo = (
    "sentence-transformers/all-roberta-large-v1"  # 用于节点和边文本编码的句向量模型。
)
batch_size = 1024  # 每次编码时处理的文本条数。


class Dataset(torch.utils.data.Dataset):  # 对分词结果做的最小 Dataset 封装。
    def __init__(
        self, input_ids=None, attention_mask=None
    ):  # 保存 token id 和 attention mask，供 DataLoader 使用。
        super().__init__()  # 初始化 PyTorch Dataset 基类。
        self.data = {  # 将各类输入张量统一收进一个字典，便于按索引读取。
            "input_ids": input_ids,  # tokenizer 输出的 token 编号。
            "att_mask": attention_mask,  # 标记真实 token 与 padding 的 attention mask。
        }

    def __len__(self):  # 返回分词后样本总数。
        return self.data["input_ids"].size(0)  # 数据集长度等于第 0 维的样本数。

    def __getitem__(self, index):  # 按索引取出一条分词后的样本。
        if isinstance(index, torch.Tensor):  # 如果索引是张量，先转成普通 Python 整数。
            index = index.item()  # 取出张量中的标量索引值。
        batch_data = dict()  # 收集当前样本的所有有效字段。
        for key in self.data.keys():  # 遍历保存的每个输入字段。
            if self.data[key] is not None:  # 跳过未提供的可选字段。
                batch_data[key] = self.data[key][
                    index
                ]  # 取出该字段在当前索引处的内容。
        return batch_data  # 返回 DataLoader 可以直接拼接的字典结构。


class Sentence_Transformer(nn.Module):  # 对预训练 Transformer 做的一层句向量编码封装。
    def __init__(self, pretrained_repo):  # 加载用于文本 embedding 的预训练模型权重。
        super(Sentence_Transformer, self).__init__()  # 初始化 nn.Module 基类。
        print(
            f"inherit model weights from {pretrained_repo}"
        )  # 打印当前加载的预训练模型名称。
        self.bert_model = AutoModel.from_pretrained(
            pretrained_repo
        )  # 实例化预训练编码器。

    def mean_pooling(
        self, model_output, attention_mask
    ):  # 将 token 级输出池化成每条文本一个向量。
        token_embeddings = model_output[0]  # 第一个返回张量保存了每个 token 的表示。
        data_type = (
            token_embeddings.dtype
        )  # 让 mask 的数据类型与 embedding 保持一致，便于后续相乘。
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).to(data_type)
        )  # 将 attention mask 扩展到 embedding 维度。
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )  # 只对真实 token 做平均，并避免分母为 0。

    def forward(self, input_ids, att_mask):  # 将一个 batch 的分词文本编码成句向量。
        bert_out = self.bert_model(
            input_ids=input_ids, attention_mask=att_mask
        )  # 用 Transformer 对输入 token 做上下文编码。
        sentence_embeddings = self.mean_pooling(
            bert_out, att_mask
        )  # 将 token 向量池化成句向量。

        sentence_embeddings = F.normalize(
            sentence_embeddings, p=2, dim=1
        )  # 对每条句向量做 L2 归一化。
        return sentence_embeddings  # 返回每条输入文本对应的归一化向量。


def load_model():  # 构建文本编码模型、分词器和运行设备。

    model = Sentence_Transformer(pretrained_repo)  # 创建句向量编码模型。
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_repo
    )  # 加载与模型匹配的 tokenizer。

    # 如果有多张 GPU，就开启简单的数据并行。
    if torch.cuda.device_count() > 1:  # 检查当前环境是否存在多张 CUDA 设备。
        print(f"Using {torch.cuda.device_count()} GPUs")  # 打印当前可用 GPU 数量。
        model = nn.DataParallel(model)  # 将编码器复制到多张 GPU 上并行执行。

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )  # 优先使用 CUDA，没有则退回 CPU。
    model.to(device)  # 将模型权重移动到目标设备上。
    model.eval()  # 切换到推理模式，关闭训练专用行为。
    return (
        model,
        tokenizer,
        device,
    )  # 返回后续文本编码所需的全部对象。


def load_text2embedding(
    model, tokenizer, device, text
):  # 将字符串列表编码成定长 embedding 矩阵。
    if len(text) == 0:  # 如果节点或边文本为空，直接走兜底分支。
        return torch.zeros((0, 1024))  # 返回一个特征维度正确的空矩阵。

    encoding = tokenizer(
        text, padding=True, truncation=True, return_tensors="pt"
    )  # 将输入文本分词并补齐成张量。
    dataset = Dataset(
        input_ids=encoding.input_ids, attention_mask=encoding.attention_mask
    )  # 将分词结果封装成 Dataset。

    # 用 DataLoader 分批处理文本，避免一次性编码过多内容。
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False
    )  # 分 batch 编码，同时保持原始顺序不变。

    # 先暂存每个 batch 的编码结果，最后再统一拼接。
    all_embeddings = []  # 保存每个 batch 的输出向量。

    # 图预处理阶段只做前向推理，不需要梯度。
    with torch.no_grad():
        for batch in dataloader:  # 逐个 batch 编码文本。
            # 将 batch 中的每个张量移动到与模型相同的设备上。
            batch = {
                key: value.to(device) for key, value in batch.items()
            }  # 把输入复制到 CPU 或 GPU。

            # 调用句向量编码器，为每条字符串得到一个向量。
            embeddings = model(
                input_ids=batch["input_ids"], att_mask=batch["att_mask"]
            )  # 通过封装后的 Transformer 做前向计算。

            # 暂存当前 batch 的输出，后面按原顺序重新拼接。
            all_embeddings.append(embeddings)  # 将当前 batch 的 embedding 加入列表。

    # 将所有 batch 的结果拼成一个 [num_texts, hidden_size] 矩阵并移回 CPU。
    all_embeddings = torch.cat(
        all_embeddings, dim=0
    ).cpu()  # 沿样本维拼接，并把结果移出 GPU。

    return all_embeddings  # 返回节点或边对应的最终 embedding 矩阵。
