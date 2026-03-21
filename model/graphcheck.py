import contextlib
import torch
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch_scatter import scatter
from model.gnn import load_gnn_model


# -----------------------------------------------------------
# Part of this code is adapted from the G-Retriever project:
# https://github.com/XiaoxinHe/G-Retriever
# He et al. (2024), "G-Retriever: Retrieval-Augmented Generation for Textual Graph Understanding and Question Answering"
# arXiv:2402.07630
# -----------------------------------------------------------


BOS = "<s>[INST]"
EOS_USER = "[/INST]"
EOS = "</s>"

IGNORE_INDEX = -100


class GraphCheck(torch.nn.Module):
    def __init__(self, args, **kwargs):
        super().__init__()
        self.max_txt_len = args.max_txt_len
        self.max_new_tokens = args.max_new_tokens

        # 按 GPU 显存自动设置每张卡可分配给 LLM 的最大显存，给系统和其他张量预留一部分空间。
        num_devices = torch.cuda.device_count()
        max_memory = {}
        for i in range(num_devices):
            total_memory = torch.cuda.get_device_properties(i).total_memory // (1024**3)
            max_memory[i] = f"{max(total_memory - 2, 2)}GiB"
        kwargs.update(
            {
                "max_memory": max_memory,
                "device_map": "auto",
                "revision": "main",
            }
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            args.llm_model_path, use_fast=False, revision=kwargs["revision"]
        )
        self.tokenizer.pad_token_id = 0
        self.tokenizer.padding_side = "left"

        # 加载底层因果语言模型，后续只把它当作冻结的生成器使用。
        model = AutoModelForCausalLM.from_pretrained(
            args.llm_model_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            **kwargs,
        )

        # 冻结 LLM 主体，只训练图相关模块。
        for name, param in model.named_parameters():
            param.requires_grad = False

        model.gradient_checkpointing_enable()

        self.model = model

        print("Finish loading LLM!!!")

        self.word_embedding = self.model.model.get_input_embeddings()

        # 方法的主要内容
        # 1. graph_encoder 负责把 claim/doc 两张图编码成节点表示。
        self.graph_encoder = load_gnn_model[args.gnn_model_name](
            in_channels=args.gnn_in_dim,
            out_channels=args.gnn_hidden_dim,
            hidden_channels=args.gnn_hidden_dim,
            num_layers=args.gnn_num_layers,
            dropout=args.gnn_dropout,
            num_heads=args.gnn_num_heads,
        ).to(self.model.device)

        # 2. projector 把图向量映射到和 LLM 词向量相同的维度，便于直接拼接到 inputs_embeds 中。
        self.projector = nn.Sequential(
            nn.Linear(args.gnn_hidden_dim, 2048),
            nn.Sigmoid(),
            nn.Linear(2048, self.word_embedding.weight.shape[1]),
        ).to(self.model.device)

        self.embed_dim = self.word_embedding.weight.shape[1]
        self.gnn_output = args.gnn_hidden_dim

    @property
    def device(self):
        return list(self.parameters())[0].device

    # 训练和推理时优先走混合精度，CPU 上则退化为普通上下文。
    def maybe_autocast(self, dtype=torch.bfloat16):
        enable_autocast = self.device != torch.device("cpu")
        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()

    # 先用 GNN 得到节点级表示，再对每张图做 mean pooling，压成 claim/doc 两个图向量。
    def encode_graphs(self, data):
        claim_kg = data["claim_kg"].to(self.model.device)
        doc_kg = data["doc_kg"].to(self.model.device)

        # 1. 得到节点级别表示
        claim_n_embeds, _ = self.graph_encoder(
            claim_kg.x, claim_kg.edge_index.long(), claim_kg.edge_attr
        )
        doc_n_embeds, _ = self.graph_encoder(
            doc_kg.x, doc_kg.edge_index.long(), doc_kg.edge_attr
        )

        # 2. 做图级别池化
        # 后面的 scatter(..., reduce='mean') 才是关键。
        # 它的作用是：
        #   1.按 batch 里的图编号，把属于同一张图的节点向量聚合起来；
        #   2.聚合方式是平均池化；
        #   3.最后每张图只保留一个向量。
        # 所以到了 encode_graphs() 的返回值阶段：
        #   1.每条样本的 claim_kg 会被压成一个向量；
        #   2.每条样本的 doc_kg 也会被压成一个向量。
        # 这说明当前实现的图注入粒度非常明确：
        #   1.不是“每个节点都变成一个 token”；
        #   2.也不是“每条边都单独送进 LLM”；
        #   3.而是“一张 claim 图一个向量，一张 doc 图一个向量”。
        if claim_kg.batch is not None:
            # batch 中可能拼了多张图，这里按图编号把节点向量聚合成图向量。
            claim_embeds = scatter(claim_n_embeds, claim_kg.batch, dim=0, reduce="mean")
        else:
            claim_embeds = claim_n_embeds.mean(dim=0, keepdim=True)

        if doc_kg.batch is not None:
            doc_embeds = scatter(doc_n_embeds, doc_kg.batch, dim=0, reduce="mean")
        else:
            doc_embeds = doc_n_embeds.mean(dim=0, keepdim=True)

        return claim_embeds, doc_embeds

    def forward(self, data):
        # 用 tokenizer 编码 batch 中每条样本的输入文本，但这里先不自动加额外特殊符号。
        texts = self.tokenizer(data["text"], add_special_tokens=False)
        # 用 tokenizer 编码 batch 中每条样本的目标标签文本，标签通常是 support / unsupport。
        labels = self.tokenizer(data["label"], add_special_tokens=False)

        # 编码结束标记 EOS，后面会接在标签末尾，表示答案结束。
        eos_tokens = self.tokenizer(EOS, add_special_tokens=False)
        # 编码用户输入结束标记，后面用它把 prompt 和答案区分开。
        eos_user_tokens = self.tokenizer(EOS_USER, add_special_tokens=False)
        # 取出 BOS 对应的词向量，后面会把它放在整个 embedding 序列最前面。
        bos_embeds = self.word_embedding(
            self.tokenizer(BOS, add_special_tokens=False, return_tensors="pt")
            .input_ids[0]
            .cuda()
        )
        # 取出 padding token 对应的词向量，后面做 batch 对齐时要重复使用它。
        pad_embeds = self.word_embedding(
            torch.tensor(self.tokenizer.pad_token_id).cuda()
        ).unsqueeze(0)

        # 用 GNN 编码 claim 图和 doc 图，得到每条样本各自的图级向量。
        claim_embeds, doc_embeds = self.encode_graphs(data)

        # 把 claim 图向量投影到 LLM 的 embedding 维度，便于和文本 embedding 直接拼接。
        claim_embeds = self.projector(claim_embeds)
        # 把 doc 图向量也投影到同一个 embedding 空间。
        doc_embeds = self.projector(doc_embeds)

        # 当前 batch 的样本数，后面会按样本逐条构造 embedding 序列。
        batch_size = len(data["id"])
        # 用来保存 batch 中每条样本最终拼好的输入 embedding 序列。
        batch_inputs_embeds = []
        # 用来保存每条样本对应的 attention mask。
        batch_attention_mask = []
        # 用来保存每条样本的监督标签序列，其中非监督位置会被 IGNORE_INDEX 屏蔽。
        batch_label_input_ids = []

        # 逐条处理 batch 中的样本，把图向量和文本 embedding 手工拼成训练输入。
        for i in range(batch_size):
            # 取出第 i 条样本的标签 token，并限制最大生成长度后再拼上 EOS。
            label_input_ids = (
                labels.input_ids[i][: self.max_new_tokens] + eos_tokens.input_ids
            )

            # 取出第 i 条样本的文本 token，截断到最大长度，再拼上用户结束标记和标签 token。
            input_ids = (
                texts.input_ids[i][: self.max_txt_len]
                + eos_user_tokens.input_ids
                + label_input_ids
            )

            # 把这条样本的 token id 序列映射成词向量序列。
            inputs_embeds = self.word_embedding(
                torch.tensor(input_ids).to(self.model.device)
            )

            # 如果 claim 图向量数量和 batch 对齐，就取出第 i 条样本对应的 claim 图向量。
            if claim_embeds.size(0) == batch_size:
                # 为了和 token embedding 在序列维度上拼接，这里补一个长度为 1 的维度。
                claim_embedding = claim_embeds[i].unsqueeze(0)
            else:
                # 如果图向量数量异常，就退化成零向量占位，避免后续拼接时报错。
                claim_embedding = (
                    torch.zeros(self.embed_dim).unsqueeze(0).to(self.model.device)
                )

            # 如果 doc 图向量数量和 batch 对齐，就取出第 i 条样本对应的 doc 图向量。
            if doc_embeds.size(0) == batch_size:
                # 同样补一个长度为 1 的序列维度，方便后续拼接。
                doc_embedding = doc_embeds[i].unsqueeze(0)
            else:
                # 如果 doc 图向量异常缺失，也用零向量占位。
                doc_embedding = (
                    torch.zeros(self.embed_dim).unsqueeze(0).to(self.model.device)
                )

            # 按顺序拼接 BOS、claim 图向量、doc 图向量和文本 embedding，形成最终输入序列。
            inputs_embeds = torch.cat(
                [bos_embeds, claim_embedding, doc_embedding, inputs_embeds], dim=0
            )

            # 把当前样本的 embedding 序列加入 batch 列表。
            batch_inputs_embeds.append(inputs_embeds)
            # 当前样本真实 token 位置都标成 1，后面 padding 的位置会标成 0。
            batch_attention_mask.append([1] * inputs_embeds.shape[0])

            # 在标签前面补齐 IGNORE_INDEX，让前面的 BOS、图向量和文本部分都不参与 loss。
            label_input_ids = [IGNORE_INDEX] * (
                inputs_embeds.shape[0] - len(label_input_ids)
            ) + label_input_ids

            # 把当前样本的监督标签序列保存起来，后面统一转成张量。
            batch_label_input_ids.append(label_input_ids)

        # 找出当前 batch 中最长的 embedding 序列长度，后面所有样本都对齐到这个长度。
        max_length = max([x.shape[0] for x in batch_inputs_embeds])

        # 再次遍历 batch，对较短样本做左侧 padding。
        for i in range(batch_size):
            # 计算第 i 条样本还差多少长度才能和 batch 内最长序列对齐。
            pad_length = max_length - batch_inputs_embeds[i].shape[0]

            # 在左侧补 pad embedding，使所有样本的 embedding 序列长度一致。
            batch_inputs_embeds[i] = torch.cat(
                [pad_embeds.repeat(pad_length, 1), batch_inputs_embeds[i]]
            )

            # padding 位置在 attention mask 中记为 0，真实位置保持为 1。
            batch_attention_mask[i] = [0] * pad_length + batch_attention_mask[i]

            # padding 位置同样不参与 loss，因此在标签序列左侧补 IGNORE_INDEX。
            batch_label_input_ids[i] = [
                IGNORE_INDEX
            ] * pad_length + batch_label_input_ids[i]

        # 把 embedding 列表堆成三维张量，形状大致是 [batch, seq_len, hidden_dim]。
        inputs_embeds = torch.stack(batch_inputs_embeds, dim=0).to(self.model.device)
        # 把 attention mask 列表转成张量，供 LLM 屏蔽 padding 位置。
        attention_mask = torch.tensor(batch_attention_mask).to(self.model.device)
        # 把标签列表转成张量，供 LLM 内部计算自回归 loss。
        label_input_ids = torch.tensor(batch_label_input_ids).to(self.model.device)

        # 在混合精度上下文中调用底层 LLM，提升显存利用率和运行效率。
        with self.maybe_autocast():
            # 直接传 inputs_embeds 而不是 input_ids，因为图信息已经被手工插入 embedding 序列。
            outputs = self.model(
                # 输入的是已经拼好图向量的 embedding 序列。
                inputs_embeds=inputs_embeds,
                # attention mask 告诉模型哪些位置是真实输入，哪些位置是 padding。
                attention_mask=attention_mask,
                # 返回 HuggingFace 标准的字典输出，便于从中取 loss。
                return_dict=True,
                # labels 只在标签 token 位置有效，其余位置都被 IGNORE_INDEX 屏蔽。
                labels=label_input_ids,
            )

        # 训练阶段只返回标量 loss，外层训练循环会负责 backward 和 optimizer.step。
        return outputs.loss

    def inference(self, data):

        # 推理时只编码输入 prompt，不再拼接真实标签。
        texts = self.tokenizer(data["text"], add_special_tokens=False)

        # 仍然需要手工准备 BOS 和 padding embedding，因为图向量依旧通过 embedding 注入。
        eos_user_tokens = self.tokenizer(EOS_USER, add_special_tokens=False)
        bos_embeds = self.word_embedding(
            self.tokenizer(BOS, add_special_tokens=False, return_tensors="pt")
            .input_ids[0]
            .cuda()
        )
        pad_embeds = self.word_embedding(
            torch.tensor(self.tokenizer.pad_token_id).cuda()
        ).unsqueeze(0)

        # 推理阶段的图处理路径与训练一致，只是最后改为 generate。
        claim_embeds, doc_embeds = self.encode_graphs(data)
        claim_embeds = self.projector(claim_embeds)
        doc_embeds = self.projector(doc_embeds)

        # data['id'] = [data['id']] if isinstance(data['id'], int) else data['id']
        batch_size = len(data["id"])

        batch_inputs_embeds = []
        batch_attention_mask = []
        for i in range(batch_size):
            input_ids = (
                texts.input_ids[i][: self.max_txt_len] + eos_user_tokens.input_ids
            )
            inputs_embeds = self.word_embedding(
                torch.tensor(input_ids).to(self.model.device)
            )

            # 若图向量数量异常，则用零向量占位，保证 embedding 序列维度不出错。
            if claim_embeds.size(0) == batch_size:
                claim_embedding = claim_embeds[i].unsqueeze(0)
            else:
                claim_embedding = (
                    torch.zeros(self.embed_dim).unsqueeze(0).to(self.model.device)
                )
            if doc_embeds.size(0) == batch_size:
                doc_embedding = doc_embeds[i].unsqueeze(0)
            else:
                doc_embedding = (
                    torch.zeros(self.embed_dim).unsqueeze(0).to(self.model.device)
                )

            inputs_embeds = torch.cat(
                [bos_embeds, claim_embedding, doc_embedding, inputs_embeds], dim=0
            )

            batch_inputs_embeds.append(inputs_embeds)
            batch_attention_mask.append([1] * inputs_embeds.shape[0])

        # 和训练阶段一样，需要先把不同长度的 embedding 序列对齐成 batch。
        max_length = max([x.shape[0] for x in batch_inputs_embeds])
        for i in range(batch_size):
            pad_length = max_length - batch_inputs_embeds[i].shape[0]
            batch_inputs_embeds[i] = torch.cat(
                [pad_embeds.repeat(pad_length, 1), batch_inputs_embeds[i]]
            )
            batch_attention_mask[i] = [0] * pad_length + batch_attention_mask[i]

        inputs_embeds = torch.stack(batch_inputs_embeds, dim=0).to(self.model.device)
        attention_mask = torch.tensor(batch_attention_mask).to(self.model.device)
        with self.maybe_autocast():
            # 这里让冻结 LLM 自回归生成标签文本，后处理阶段再从文本中解析 support / unsupport。
            outputs = self.model.generate(
                inputs_embeds=inputs_embeds,
                max_new_tokens=self.max_new_tokens,
                attention_mask=attention_mask,
                use_cache=True,
            )
        pred = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        print(pred)
        return {
            "id": data["id"],
            "pred": pred,
            "label": data["label"],
            "text": data["text"],
        }

    def print_trainable_params(self):
        trainable_params = 0
        all_param = 0

        for _, param in self.named_parameters():
            num_params = param.numel()

            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params

        # 用来确认当前训练到底只更新了多少参数。
        return trainable_params, all_param
