import pandas as pd
import torch
from torch.utils.data import Dataset
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PATH = f"{project_root}"


def get_dataset(dataset_name):
    # 读取指定数据集对应的 pkl 文件，其中保存了文本、KG 三元组和标签等元数据。
    result_df = pd.read_pickle(f"{PATH}/extracted_KG/{dataset_name}/{dataset_name}.pkl")
    # 取出原始文档文本。
    docs = result_df["doc_text"].values
    # 取出待核查的 claim 文本。
    claims = result_df["claim_text"].values
    # 取出文档侧的 KG 三元组文本表示。
    doc_kgs = result_df["doc_kg"].values
    # 取出 claim 侧的 KG 三元组文本表示。
    claim_kgs = result_df["claim_kg"].values
    # 取出监督标签，通常是 0/1。
    labels = result_df["label"].values
    # 返回后续构造数据集对象所需的全部字段。
    return docs, claims, doc_kgs, claim_kgs, labels


class KGDataset(Dataset):
    def __init__(self, dataset_name):
        # 初始化父类 Dataset。
        super().__init__()
        # 记录当前数据集名称，后续读取图文件和 split 文件时会用到。
        self.dataset_name = dataset_name
        # 一次性读入当前数据集的文本、KG 和标签元数据。
        self.docs, self.claims, self.doc_kgs, self.claim_kgs, self.labels = get_dataset(
            self.dataset_name
        )
        # 定义训练和推理时共用的指令模板。
        self.prompt = "Question: Does the Document support the Claim? Please Answer in one word in the form of 'support' or 'unsupport'.\n\n"

    def __len__(self):
        # 返回数据集样本总数，DataLoader 会依赖这个长度做索引访问。
        return len(self.docs)

    def __getitem__(self, index):
        """
        按索引取出一条完整的样本
        这几行代码非常关键，因为它说明一条训练样本实际上由三部分组成：
            1.文本输入：Claim + Document 拼成的 prompt；
            2.图输入：claim_kg.pt 和 doc_kg.pt 两个图对象；
            3.标签：最终被转成自然语言形式的 support / unsupport。
            也就是说，GraphCheck 不是一个“只吃图”的模型，也不是一个“只吃文本”的模型，而是文本和图同时输入。
        """
        # 根据索引取出当前样本对应的文档、claim、KG 文本和标签。
        doc, claim, doc_kgs_text, claim_kgs_text, label = (
            self.docs[index],
            self.claims[index],
            self.doc_kgs[index],
            self.claim_kgs[index],
            self.labels[index],
        )
        # 将 prompt、claim 和 document 拼成最终送入 LLM 的文本输入。
        text = f"{self.prompt}\nClaim: {claim}\nDocument: {doc}"
        # 读取当前样本对应的 claim 图对象。
        claim_kg = torch.load(
            f"{PATH}/extracted_KG/{self.dataset_name}/graphs/claim/{index}.pt"
        )
        # 读取当前样本对应的 doc 图对象。
        doc_kg = torch.load(
            f"{PATH}/extracted_KG/{self.dataset_name}/graphs/doc/{index}.pt"
        )
        # 将数值标签映射成训练时需要的自然语言标签。
        if label == 1:
            label = "support"
        else:
            label = "unsupport"

        # 返回一条完整样本，里面同时包含文本输入、两张图以及监督标签。
        # 模型后面拿到的不是单独的一个 tensor，而是一份“多模态样本包”
        return {
            "id": index,
            "label": label,
            "claim_kg": claim_kg,
            "doc_kg": doc_kg,
            "claim_kg_text": claim_kgs_text,
            "doc_kg_text": doc_kgs_text,
            "text": text,
            "index": index,
            "dataset": self.dataset_name,
        }

    def get_idx_split(self):
        """
        读取预处理阶段保存的划分文件
        :return:
        """
        # 读取预处理阶段保存的训练集索引。
        with open(
            f"{PATH}/extracted_KG/{self.dataset_name}/split/train_indices.txt", "r"
        ) as file:
            train_indices = [int(line.strip()) for line in file]

        # 读取预处理阶段保存的验证集索引。
        with open(
            f"{PATH}/extracted_KG/{self.dataset_name}/split/val_indices.txt", "r"
        ) as file:
            val_indices = [int(line.strip()) for line in file]

        # 读取预处理阶段保存的测试集索引。
        with open(
            f"{PATH}/extracted_KG/{self.dataset_name}/split/test_indices.txt", "r"
        ) as file:
            test_indices = [int(line.strip()) for line in file]

        # 返回一个字典，供 train.py 按照索引构造 train/val/test 子集。
        return {"train": train_indices, "val": val_indices, "test": test_indices}


if __name__ == "__main__":
    dataset = KGDataset()
    data = dataset[0]
    for k, v in data.items():
        print(f"{k}: {v}")
