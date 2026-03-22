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


class GraphCheck(torch.nn.Module):  # 定义 GraphCheck 主模型，负责把图信息接入冻结 LLM。
    def __init__(self, args, **kwargs):  # 初始化模型、LLM、GNN 和投影头。
        super().__init__()  # 先初始化父类 torch.nn.Module。
        self.max_txt_len = args.max_txt_len  # 保存文本最大截断长度。
        self.max_new_tokens = args.max_new_tokens  # 保存生成标签时允许的最大 token 数。

        # 按 GPU 显存自动设置每张卡可分配给 LLM 的最大显存，给系统和其他张量预留一部分空间。
        num_devices = torch.cuda.device_count()  # 统计当前可见的 GPU 数量。
        max_memory = {}  # 为 HuggingFace 的 device_map 构造每张卡的显存上限字典。
        for i in range(num_devices):  # 逐张卡读取显存容量并设置可用上限。
            total_memory = torch.cuda.get_device_properties(i).total_memory // (1024**3)  # 把显存总量换算成 GiB。
            max_memory[i] = f"{max(total_memory - 2, 2)}GiB"  # 预留约 2GiB 给系统和其他张量，至少保留 2GiB 配额。
        kwargs.update(  # 把分配策略和 revision 一并写入加载 LLM 时的参数。
            {
                "max_memory": max_memory,  # 指定每张 GPU 可分配给模型的显存上限。
                "device_map": "auto",  # 让 HuggingFace 自动把模型切到合适的设备上。
                "revision": "main",  # 默认使用仓库的 main 分支权重。
            }
        )

        self.tokenizer = AutoTokenizer.from_pretrained(  # 加载与底层 LLM 对应的 tokenizer。
            args.llm_model_path, use_fast=False, revision=kwargs["revision"]
        )
        self.tokenizer.pad_token_id = 0  # 显式把 pad token id 设为 0，便于后续左侧 padding。
        self.tokenizer.padding_side = "left"  # 设为左侧 padding，以适配自回归模型的输入习惯。

        # 加载底层因果语言模型，后续只把它当作冻结的生成器使用。
        model = AutoModelForCausalLM.from_pretrained(  # 从 HuggingFace 或本地目录加载 Causal LM。
            args.llm_model_path,
            torch_dtype=torch.float16,  # 用 float16 降低显存占用。
            low_cpu_mem_usage=True,  # 尽量减少加载时的 CPU 内存峰值。
            **kwargs,
        )

        # 冻结 LLM 主体，只训练图相关模块。
        for name, param in model.named_parameters():  # 遍历 LLM 的全部参数。
            param.requires_grad = False  # 对每个参数关闭梯度，避免训练中更新 LLM。

        model.gradient_checkpointing_enable()  # 开启梯度检查点，进一步节省显存。

        self.model = model  # 把底层 LLM 挂到当前模型实例上。

        print("Finish loading LLM!!!")  # 打印提示，说明 LLM 已成功加载。

        self.word_embedding = self.model.model.get_input_embeddings()  # 取出底层 LLM 的词嵌入层，后面要直接把 token 变成 embedding。

        # 方法的主要内容
        # 1. graph_encoder 负责把 claim/doc 两张图编码成节点表示。
        self.graph_encoder = load_gnn_model[args.gnn_model_name](  # 根据配置选择具体的 GNN 实现并实例化。
            in_channels=args.gnn_in_dim,  # 输入维度，对应节点/边文本向量维度。
            out_channels=args.gnn_hidden_dim,  # 最终输出维度，这里与隐藏维度保持一致。
            hidden_channels=args.gnn_hidden_dim,  # 隐藏层维度。
            num_layers=args.gnn_num_layers,  # GNN 堆叠层数。
            dropout=args.gnn_dropout,  # GNN 内部 dropout 比例。
            num_heads=args.gnn_num_heads,  # 图注意力头数。
        ).to(self.model.device)  # 把 GNN 放到和 LLM 主设备一致的位置。

        # 2. projector 把图向量映射到和 LLM 词向量相同的维度，便于直接拼接到 inputs_embeds 中。
        self.projector = nn.Sequential(  # 用一个两层 MLP 作为投影头。
            nn.Linear(args.gnn_hidden_dim, 2048),  # 先把图向量从 GNN 维度映射到中间维度 2048。
            nn.Sigmoid(),  # 加入非线性，使投影变换不只是单层线性映射。
            nn.Linear(2048, self.word_embedding.weight.shape[1]),  # 再映射到 LLM 词向量维度。
        ).to(self.model.device)  # 把投影头也放到 LLM 所在设备。

        self.embed_dim = self.word_embedding.weight.shape[1]  # 缓存 LLM embedding 维度，后面创建零向量占位时要用。
        self.gnn_output = args.gnn_hidden_dim  # 记录 GNN 输出维度，便于外部查看配置。

    @property  # 把 device 写成属性形式，方便统一查询当前模型所在设备。
    def device(self):  # 返回当前模型参数所在的设备。
        return list(self.parameters())[0].device  # 默认取第一个参数的设备作为模型设备。

    # 训练和推理时优先走混合精度，CPU 上则退化为普通上下文。
    def maybe_autocast(self, dtype=torch.bfloat16):  # 返回一个可直接 with 使用的上下文管理器。
        enable_autocast = self.device != torch.device("cpu")  # 只有在 GPU 环境下才启用 autocast。
        if enable_autocast:  # 如果当前设备不是 CPU。
            return torch.cuda.amp.autocast(dtype=dtype)  # 返回 CUDA 混合精度上下文。
        else:  # 如果当前运行在 CPU 上。
            return contextlib.nullcontext()  # 返回空上下文，等价于不做混合精度。

    # 先用 GNN 得到节点级表示，再对每张图做 mean pooling，压成 claim/doc 两个图向量。
    def encode_graphs(self, data):  # 编码 batch 中的 claim 图和 doc 图。
        claim_kg = data["claim_kg"].to(self.model.device)  # 取出 claim 图并搬到模型设备。
        doc_kg = data["doc_kg"].to(self.model.device)  # 取出 doc 图并搬到模型设备。

        # 1. 得到节点级别表示
        claim_n_embeds, _ = self.graph_encoder(  # 用 GNN 编码 claim 图，得到每个节点的新表示。
            claim_kg.x, claim_kg.edge_index.long(), claim_kg.edge_attr
        )
        doc_n_embeds, _ = self.graph_encoder(  # 用同一个 GNN 编码 doc 图，得到每个节点的新表示。
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
        if claim_kg.batch is not None:  # 如果 claim 图来自一个 batched PyG 对象。
            claim_embeds = scatter(claim_n_embeds, claim_kg.batch, dim=0, reduce="mean")  # 按图编号做平均池化，得到每张 claim 图的图级向量。
        else:  # 如果当前只有一张图，没有显式的 batch 向量。
            claim_embeds = claim_n_embeds.mean(dim=0, keepdim=True)  # 直接对节点维做平均，得到单张图的图级向量。

        if doc_kg.batch is not None:  # 如果 doc 图来自 batched PyG 对象。
            doc_embeds = scatter(doc_n_embeds, doc_kg.batch, dim=0, reduce="mean")  # 按图编号做平均池化，得到每张 doc 图的图级向量。
        else:  # 如果当前只有一张 doc 图。
            doc_embeds = doc_n_embeds.mean(dim=0, keepdim=True)  # 直接对节点做平均，得到单张图的图级向量。

        return claim_embeds, doc_embeds  # 返回 batch 内每条样本对应的 claim/doc 图向量。

    def forward(self, data):  # 训练阶段的前向传播，最终返回一个标量 loss。
        # 用 tokenizer 编码 batch 中每条样本的输入文本，但这里先不自动加额外特殊符号。
        texts = self.tokenizer(data["text"], add_special_tokens=False)  # 把文本 prompt 编成 token id 序列。
        # 用 tokenizer 编码 batch 中每条样本的目标标签文本，标签通常是 support / unsupport。
        labels = self.tokenizer(data["label"], add_special_tokens=False)  # 把监督标签文本也编码成 token id 序列。

        # 编码结束标记 EOS，后面会接在标签末尾，表示答案结束。
        eos_tokens = self.tokenizer(EOS, add_special_tokens=False)  # 单独编码答案结束标记。
        # 编码用户输入结束标记，后面用它把 prompt 和答案区分开。
        eos_user_tokens = self.tokenizer(EOS_USER, add_special_tokens=False)  # 单独编码用户输入结束标记。
        # 取出 BOS 对应的词向量，后面会把它放在整个 embedding 序列最前面。
        bos_embeds = self.word_embedding(  # 先把 BOS 文本转成 token id，再取对应的词向量。
            self.tokenizer(BOS, add_special_tokens=False, return_tensors="pt")
            .input_ids[0]
            .cuda()
        )
        # 取出 padding token 对应的词向量，后面做 batch 对齐时要重复使用它。
        pad_embeds = self.word_embedding(  # 把 pad token id 映射成一个 embedding 向量。
            torch.tensor(self.tokenizer.pad_token_id).cuda()
        ).unsqueeze(0)  # 补一个序列维，便于后面 repeat 和拼接。

        # 用 GNN 编码 claim 图和 doc 图，得到每条样本各自的图级向量。
        claim_embeds, doc_embeds = self.encode_graphs(data)  # 先把两类图压成图级向量。

        # 把 claim 图向量投影到 LLM 的 embedding 维度，便于和文本 embedding 直接拼接。
        claim_embeds = self.projector(claim_embeds)  # 把 claim 图向量变换到 LLM embedding 空间。
        # 把 doc 图向量也投影到同一个 embedding 空间。
        doc_embeds = self.projector(doc_embeds)  # 把 doc 图向量也投影到同一个空间。

        # 当前 batch 的样本数，后面会按样本逐条构造 embedding 序列。
        batch_size = len(data["id"])  # 通过 id 列表长度得到当前 batch 大小。
        # 用来保存 batch 中每条样本最终拼好的输入 embedding 序列。
        batch_inputs_embeds = []  # 存放每条样本的最终输入 embedding。
        # 用来保存每条样本对应的 attention mask。
        batch_attention_mask = []  # 存放每条样本的 attention mask。
        # 用来保存每条样本的监督标签序列，其中非监督位置会被 IGNORE_INDEX 屏蔽。
        batch_label_input_ids = []  # 存放每条样本最终对齐后的 labels。

        # 逐条处理 batch 中的样本，把图向量和文本 embedding 手工拼成训练输入。
        for i in range(batch_size):  # 逐个样本构造训练时真正送入 LLM 的序列。
            # 取出第 i 条样本的标签 token，并限制最大生成长度后再拼上 EOS。
            label_input_ids = (  # 这一段是真实答案对应的 token 序列。
                labels.input_ids[i][: self.max_new_tokens] + eos_tokens.input_ids
            )

            # 取出第 i 条样本的文本 token，截断到最大长度，再拼上用户结束标记和标签 token。
            input_ids = (  # 训练时输入里包含 prompt、分隔标记和真实标签前缀。
                texts.input_ids[i][: self.max_txt_len]
                + eos_user_tokens.input_ids
                + label_input_ids
            )

            # 把这条样本的 token id 序列映射成词向量序列。
            inputs_embeds = self.word_embedding(  # 直接把 token id 映射成 LLM 的输入 embedding。
                torch.tensor(input_ids).to(self.model.device)
            )

            # 如果 claim 图向量数量和 batch 对齐，就取出第 i 条样本对应的 claim 图向量。
            if claim_embeds.size(0) == batch_size:  # 正常情况下，图向量数应与 batch 大小一致。
                claim_embedding = claim_embeds[i].unsqueeze(0)  # 取出当前样本的 claim 图向量并补序列维。
            else:  # 如果图向量数量异常。
                claim_embedding = (  # 用零向量占位，避免后续拼接时报维度错误。
                    torch.zeros(self.embed_dim).unsqueeze(0).to(self.model.device)
                )

            # 如果 doc 图向量数量和 batch 对齐，就取出第 i 条样本对应的 doc 图向量。
            if doc_embeds.size(0) == batch_size:  # 正常情况下，doc 图向量数也应与 batch 对齐。
                doc_embedding = doc_embeds[i].unsqueeze(0)  # 取出当前样本的 doc 图向量并补序列维。
            else:  # 如果 doc 图向量数量异常。
                doc_embedding = (  # 同样退化成零向量占位。
                    torch.zeros(self.embed_dim).unsqueeze(0).to(self.model.device)
                )

            # 按顺序拼接 BOS、claim 图向量、doc 图向量和文本 embedding，形成最终输入序列。
            inputs_embeds = torch.cat(  # 最终顺序是 BOS -> claim 图 -> doc 图 -> 文本+标签 embedding。
                [bos_embeds, claim_embedding, doc_embedding, inputs_embeds], dim=0
            )

            # 把当前样本的 embedding 序列加入 batch 列表。
            batch_inputs_embeds.append(inputs_embeds)  # 保存当前样本的最终 embedding 序列。
            # 当前样本真实 token 位置都标成 1，后面 padding 的位置会标成 0。
            batch_attention_mask.append([1] * inputs_embeds.shape[0])  # 先把当前真实长度全部标记为可见。

            # 在标签前面补齐 IGNORE_INDEX，让前面的 BOS、图向量和文本部分都不参与 loss。
            label_input_ids = [IGNORE_INDEX] * (  # 把非监督位置全设为 -100，交给 HF 内部忽略。
                inputs_embeds.shape[0] - len(label_input_ids)
            ) + label_input_ids

            # 把当前样本的监督标签序列保存起来，后面统一转成张量。
            batch_label_input_ids.append(label_input_ids)  # 把当前样本的 label 序列加入 batch。

        # 找出当前 batch 中最长的 embedding 序列长度，后面所有样本都对齐到这个长度。
        max_length = max([x.shape[0] for x in batch_inputs_embeds])  # 统计 batch 内最长序列长度。

        # 再次遍历 batch，对较短样本做左侧 padding。
        for i in range(batch_size):  # 逐个样本补齐到统一长度。
            # 计算第 i 条样本还差多少长度才能和 batch 内最长序列对齐。
            pad_length = max_length - batch_inputs_embeds[i].shape[0]  # 需要在左侧补多少个 pad。

            # 在左侧补 pad embedding，使所有样本的 embedding 序列长度一致。
            batch_inputs_embeds[i] = torch.cat(  # 用 pad embedding 在左侧补齐序列长度。
                [pad_embeds.repeat(pad_length, 1), batch_inputs_embeds[i]]
            )

            # padding 位置在 attention mask 中记为 0，真实位置保持为 1。
            batch_attention_mask[i] = [0] * pad_length + batch_attention_mask[i]  # 为左侧 pad 位置补 0。

            # padding 位置同样不参与 loss，因此在标签序列左侧补 IGNORE_INDEX。
            batch_label_input_ids[i] = [  # 让左侧 padding 位置在 loss 里同样被忽略。
                IGNORE_INDEX
            ] * pad_length + batch_label_input_ids[i]

        # 把 embedding 列表堆成三维张量，形状大致是 [batch, seq_len, hidden_dim]。
        inputs_embeds = torch.stack(batch_inputs_embeds, dim=0).to(self.model.device)  # 把样本列表堆成 batch 张量。
        # 把 attention mask 列表转成张量，供 LLM 屏蔽 padding 位置。
        attention_mask = torch.tensor(batch_attention_mask).to(self.model.device)  # 转成 attention mask 张量。
        # 把标签列表转成张量，供 LLM 内部计算自回归 loss。
        label_input_ids = torch.tensor(batch_label_input_ids).to(self.model.device)  # 转成 labels 张量。

        # 在混合精度上下文中调用底层 LLM，提升显存利用率和运行效率。
        with self.maybe_autocast():  # 根据设备自动选择是否启用 autocast。
            # 直接传 inputs_embeds 而不是 input_ids，因为图信息已经被手工插入 embedding 序列。
            outputs = self.model(  # 调用底层 Causal LM，直接让它内部计算 loss。
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
        return outputs.loss  # 直接把 HuggingFace 计算好的 loss 返回给外部训练循环。

    def inference(self, data):  # 推理阶段的前向逻辑，返回预测文本而不是 loss。
        """
        inference 在测试/评估阶段使用，有两个调用位置：
            1. train.py:203 — 训练完成后测试
                # 加载 best checkpoint 后，在测试集上评估
                model = _reload_best_model(model, args)
                model.eval()
                for step, batch in enumerate(test_loader):
                    output = model.inference(batch)  # 测试集预测
            2. inference.py:53 — 独立推理脚本
                # 单独运行推理，不经过训练
                model = _reload_best_model(model, args)
                model.eval()
                for _, batch in enumerate(test_loader):
                    output = model.inference(batch)  # 加载已训练模型做预测
        """

        # 推理时只编码输入 prompt，不再拼接真实标签。
        texts = self.tokenizer(data["text"], add_special_tokens=False)  # 把待预测的输入文本编码成 token 序列。

        # 仍然需要手工准备 BOS 和 padding embedding，因为图向量依旧通过 embedding 注入。
        eos_user_tokens = self.tokenizer(EOS_USER, add_special_tokens=False)  # 编码用户输入结束标记，用于界定 prompt 结束。
        bos_embeds = self.word_embedding(  # 取出 BOS 对应的 embedding，放在整条序列最前面。
            self.tokenizer(BOS, add_special_tokens=False, return_tensors="pt")
            .input_ids[0]
            .cuda()
        )
        pad_embeds = self.word_embedding(  # 取出 pad token 的 embedding，供 batch 对齐复用。
            torch.tensor(self.tokenizer.pad_token_id).cuda()
        ).unsqueeze(0)  # 补一个序列维，便于 repeat。

        # 推理阶段的图处理路径与训练一致，只是最后改为 generate。
        claim_embeds, doc_embeds = self.encode_graphs(data)  # 先把图编码成图级向量。
        claim_embeds = self.projector(claim_embeds)  # 把 claim 图向量投影到 LLM embedding 空间。
        doc_embeds = self.projector(doc_embeds)  # 把 doc 图向量也投影到同一个空间。

        # data['id'] = [data['id']] if isinstance(data['id'], int) else data['id']
        batch_size = len(data["id"])  # 取当前推理 batch 的样本数。

        batch_inputs_embeds = []  # 保存每条样本的推理输入 embedding。
        batch_attention_mask = []  # 保存每条样本的推理 attention mask。
        for i in range(batch_size):  # 逐条构造推理时的输入序列。
            input_ids = (  # 推理时只保留 prompt，不拼真实标签。
                texts.input_ids[i][: self.max_txt_len] + eos_user_tokens.input_ids
            )
            inputs_embeds = self.word_embedding(  # 把 prompt token 转成词向量。
                torch.tensor(input_ids).to(self.model.device)
            )

            # 若图向量数量异常，则用零向量占位，保证 embedding 序列维度不出错。
            if claim_embeds.size(0) == batch_size:  # 正常情况下 claim 图向量数量与 batch 对齐。
                claim_embedding = claim_embeds[i].unsqueeze(0)  # 取出当前样本的 claim 图向量并补序列维。
            else:  # 如果 claim 图向量数量异常。
                claim_embedding = (  # 用零向量占位，避免拼接失败。
                    torch.zeros(self.embed_dim).unsqueeze(0).to(self.model.device)
                )
            if doc_embeds.size(0) == batch_size:  # 正常情况下 doc 图向量数量也与 batch 对齐。
                doc_embedding = doc_embeds[i].unsqueeze(0)  # 取出当前样本的 doc 图向量并补序列维。
            else:  # 如果 doc 图向量数量异常。
                doc_embedding = (  # 同样退化成零向量占位。
                    torch.zeros(self.embed_dim).unsqueeze(0).to(self.model.device)
                )

            inputs_embeds = torch.cat(  # 推理输入同样按 BOS -> claim 图 -> doc 图 -> 文本顺序拼接。
                [bos_embeds, claim_embedding, doc_embedding, inputs_embeds], dim=0
            )

            batch_inputs_embeds.append(inputs_embeds)  # 保存当前样本的 embedding 序列。
            batch_attention_mask.append([1] * inputs_embeds.shape[0])  # 当前真实 token 全部标为可见。

        # 和训练阶段一样，需要先把不同长度的 embedding 序列对齐成 batch。
        max_length = max([x.shape[0] for x in batch_inputs_embeds])  # 统计当前 batch 内最长序列长度。
        for i in range(batch_size):  # 逐条样本补齐到统一长度。
            pad_length = max_length - batch_inputs_embeds[i].shape[0]  # 计算当前样本需要补多少 pad。
            batch_inputs_embeds[i] = torch.cat(  # 在左侧补 pad embedding。
                [pad_embeds.repeat(pad_length, 1), batch_inputs_embeds[i]]
            )
            batch_attention_mask[i] = [0] * pad_length + batch_attention_mask[i]  # 为左侧 pad 位置补 attention mask=0。

        inputs_embeds = torch.stack(batch_inputs_embeds, dim=0).to(self.model.device)  # 把推理样本列表堆成 batch embedding 张量。
        attention_mask = torch.tensor(batch_attention_mask).to(self.model.device)  # 把 attention mask 列表转成张量。
        with self.maybe_autocast():  # 推理阶段同样按设备情况启用混合精度。
            # 这里让冻结 LLM 自回归生成标签文本，后处理阶段再从文本中解析 support / unsupport。
            outputs = self.model.generate(  # 调用底层 LLM 的 generate 生成预测文本。
                inputs_embeds=inputs_embeds,
                max_new_tokens=self.max_new_tokens,
                attention_mask=attention_mask,
                use_cache=True,
            )
        pred = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)  # 把生成出的 token 序列解码成字符串。
        print(pred)  # 打印当前 batch 的原始文本预测结果，便于调试查看。
        return {  # 返回用于评估和落盘的结果字典。
            "id": data["id"],  # 返回样本 id。
            "pred": pred,  # 返回模型生成的预测文本。
            "label": data["label"],  # 一并返回真实标签，便于后处理评估。
            "text": data["text"],  # 一并返回原始输入文本，便于人工检查。
        }

    def print_trainable_params(self):  # 统计当前模型中可训练参数量与总参数量。
        trainable_params = 0  # 初始化可训练参数计数器。
        all_param = 0  # 初始化总参数计数器。

        for _, param in self.named_parameters():  # 遍历当前模型的所有参数。
            num_params = param.numel()  # 计算当前参数张量的元素个数。

            all_param += num_params  # 把当前参数量累加到总参数量里。
            if param.requires_grad:  # 如果当前参数是可训练的。
                trainable_params += num_params  # 就把它累加到可训练参数统计中。

        # 用来确认当前训练到底只更新了多少参数。
        return trainable_params, all_param  # 返回可训练参数量和总参数量。
