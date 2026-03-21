import os
import gc
from tqdm import tqdm
import torch
import json
import pandas as pd
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from model import resolve_llm_model_path
from dataset.utils.dataset import KGDataset
from src.config import parse_args
from src.ckpt import _save_checkpoint, _reload_best_model
from dataset.utils.collate import collate_fn
from src.utils import seed_everything, adjust_learning_rate, get_accuracy
from model.graphcheck import GraphCheck


# -----------------------------------------------------------
# Part of this code is adapted from the G-Retriever project:
# https://github.com/XiaoxinHe/G-Retriever
# He et al. (2024), "G-Retriever: Retrieval-Augmented Generation for Textual Graph Understanding and Question Answering"
# arXiv:2402.07630
# -----------------------------------------------------------


def main(args):
    # 读取随机种子配置，保证训练过程尽量可复现。
    seed = args.seed
    # 设置 Python、PyTorch 等相关随机种子。
    seed_everything(seed=args.seed)
    # 打印当前训练配置，方便排查参数问题。
    print(args)

    # 构造完整数据集对象，里面包含文本、图和标签。
    dataset = KGDataset(args.train_dataset)
    # 读取预处理阶段保存的 train/val/test 索引。
    idx_split = dataset.get_idx_split()

    # 按训练索引取出训练子集。
    train_dataset = [dataset[i] for i in idx_split["train"]]
    # 按验证索引取出验证子集。
    val_dataset = [dataset[i] for i in idx_split["val"]]
    # 按测试索引取出测试子集。
    test_dataset = [dataset[i] for i in idx_split["test"]]

    # 构造训练集 DataLoader，训练时开启 shuffle 打乱样本顺序。
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        drop_last=True,
        pin_memory=True,
        shuffle=True,
        collate_fn=collate_fn,
    )
    # 构造验证集 DataLoader，验证时不打乱顺序。
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        drop_last=False,
        pin_memory=True,
        shuffle=False,
        collate_fn=collate_fn,
    )
    # 构造测试集 DataLoader，评估时使用单独的 batch 大小配置。
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.eval_batch_size,
        drop_last=False,
        pin_memory=True,
        shuffle=False,
        collate_fn=collate_fn,
    )

    # 根据模型名解析出默认的 HuggingFace repo id，或保留用户传入的本地模型目录。
    args.llm_model_path = resolve_llm_model_path(
        args.llm_model_name, args.llm_model_path
    )
    # 构建 GraphCheck 模型，内部会加载冻结的 LLM、GNN 和 projector。
    model = GraphCheck(args=args) # 模型重点

    # 只收集需要训练的参数，冻结的 LLM 参数不会进入优化器。
    params = [p for _, p in model.named_parameters() if p.requires_grad]
    # 使用 AdamW 优化器更新可训练模块。
    optimizer = torch.optim.AdamW(
        [
            {"params": params, "lr": args.lr, "weight_decay": args.wd},
        ],
        betas=(0.9, 0.95),
    )
    # 统计可训练参数量和总参数量，方便确认是否真的只训练部分模块。
    trainable_params, all_param = model.print_trainable_params()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

    # 总训练步数 = epoch 数 * 每个 epoch 的 batch 数。
    num_training_steps = args.num_epochs * len(train_loader)
    # 创建训练进度条。
    progress_bar = tqdm(range(num_training_steps))
    # 用于记录当前最优验证损失，初始化为正无穷。
    best_val_loss = float("inf")

    # 逐个 epoch 进行训练。
    for epoch in range(args.num_epochs):
        # 切换到训练模式。
        model.train()
        # epoch_loss 记录当前 epoch 的总损失，accum_loss 用于梯度累积场景下的局部统计。
        epoch_loss, accum_loss = 0.0, 0.0

        # 遍历当前 epoch 中的每一个训练 batch。
        for step, batch in enumerate(train_loader):
            # 清空上一轮迭代残留的梯度。
            optimizer.zero_grad()
            # 前向计算，返回当前 batch 的训练损失。
            loss = model(batch)
            # 反向传播，计算各参数梯度。
            loss.backward()

            # 对梯度做裁剪，避免梯度过大导致训练不稳定。
            clip_grad_norm_(optimizer.param_groups[0]["params"], 0.1)

            # 按配置的梯度步数调整学习率。
            if (step + 1) % args.grad_steps == 0:
                adjust_learning_rate(
                    optimizer.param_groups[0],
                    args.lr,
                    step / len(train_loader) + epoch,
                    args,
                )

            # 用当前梯度更新一次可训练参数。
            optimizer.step()
            # 累加 epoch 总损失和当前统计窗口内的损失。
            epoch_loss, accum_loss = epoch_loss + loss.item(), accum_loss + loss.item()

            # 到达指定梯度步数后，读取当前学习率并清空窗口损失统计。
            if (step + 1) % args.grad_steps == 0:
                lr = optimizer.param_groups[0]["lr"]
                accum_loss = 0.0

            # 更新训练进度条。
            progress_bar.update(1)

        # 打印当前 epoch 的平均训练损失。
        print(
            f"Epoch: {epoch}|{args.num_epochs}: Train Loss (Epoch Mean): {epoch_loss / len(train_loader)}"
        )

        # 初始化当前 epoch 的验证损失。
        val_loss = 0.0
        # 预留评估输出列表，当前版本中未实际使用。
        eval_output = []
        # 切换到评估模式。
        model.eval()
        # 验证阶段不需要计算梯度。
        with torch.no_grad():
            # 遍历验证集的每一个 batch。
            for step, batch in enumerate(val_loader):
                # 直接复用 forward 计算验证损失。
                loss = model(batch)
                val_loss += loss.item()
            # 计算平均验证损失。
            val_loss = val_loss / len(val_loader)
            print(f"Epoch: {epoch}|{args.num_epochs}: Val Loss: {val_loss}")

        # 如果当前验证损失更优，就保存 best checkpoint。
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            _save_checkpoint(model, optimizer, epoch, args, is_best=True)
            best_epoch = epoch

        # 打印当前 epoch 和历史最优验证结果。
        print(
            f"Epoch {epoch} Val Loss {val_loss} Best Val Loss {best_val_loss} Best Epoch {best_epoch}"
        )

        # 如果连续若干个 epoch 没有提升，就触发早停。
        if epoch - best_epoch >= args.patience:
            print(f"Early stop at epoch {epoch}")
            break

    # 训练结束后清理部分 CUDA 显存统计信息。
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()

    # 创建测试结果保存目录。
    os.makedirs(f"{args.output_dir}/{args.project}", exist_ok=True)
    # 定义测试集预测结果输出文件路径。
    path = path = f"{args.output_dir}/{args.project}/validation.csv"
    print(f"path: {path}")

    # 重新加载验证集上表现最好的模型参数。
    model = _reload_best_model(model, args)
    # 切换到评估模式。
    model.eval()
    # 创建测试阶段进度条。
    progress_bar_test = tqdm(range(len(test_loader)))
    # 将测试集逐条预测结果写入文件。
    with open(path, "w") as f:
        for step, batch in enumerate(test_loader):
            with torch.no_grad():
                # 生成当前 batch 的预测结果。
                output = model.inference(batch)
                # 转成 DataFrame 便于逐行写出。
                df = pd.DataFrame(output)
                for _, row in df.iterrows():
                    f.write(json.dumps(dict(row)) + "\n")
            progress_bar_test.update(1)

    # 读取预测文件并计算 balanced accuracy。
    bacc = get_accuracy(path)
    print(f"Test BAcc: {bacc}")


if __name__ == "__main__":
    # 解析命令行参数。
    args = parse_args()

    # 执行训练主流程。
    main(args)
    # 脚本结束前再次清理显存和 Python 垃圾对象。
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    gc.collect()
