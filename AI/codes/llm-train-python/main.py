import sys
from pathlib import Path

import torch

from model.GPTModel import GPTModel
from tool.data_loader import create_dataloader_v1

# 两档配置：
#   normal：GPT-2 124M 完整配置（12 层），跑完整训练并绘制 loss 曲线。
#   small ：小配置（3 层），逐 step 打印 loss，用于快速冒烟和与 C++ 版逐 step 对齐。
# 默认跑 normal，显式传入命令行参数 small 才跑小配置：`python main.py small`。
GPT_CONFIG_124M = {
    'vocab_size': 50257,
    'context_length': 256,
    'emb_dim': 768,
    'n_heads': 12,
    'n_layers': 12,
    'drop_rate': 0.1,
    'qkv_bias': False,
}

SMALL_CONFIG = {
    'vocab_size': 1360,
    'context_length': 64,
    'emb_dim': 128,
    'n_heads': 4,
    'n_layers': 3,
    'drop_rate': 0.0,
    'qkv_bias': False,
}

profile = sys.argv[1] if len(sys.argv) > 1 else 'normal'
is_small = (profile == 'small')

torch.manual_seed(123)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BASE_DIR = Path(__file__).resolve().parent
file_path = BASE_DIR / "train_data" / "the-verdict.txt"
with open(file_path, "r", encoding="utf-8") as file:
    text_data = file.read()


def run_small():
    """小配置：逐 step 打印 loss，与 C++ `train_gpt <backend> small` 对齐。

    直接读取压缩词表版 token id（the_verdict_train_ids_small.txt，id 已重映射到 0..1359），
    把 vocab 从 50257 缩到 1360，CPU 上最重的输出层计算量降到约 1/37，单步快几十倍。
    """
    from torch.utils.data import DataLoader, TensorDataset

    cfg = SMALL_CONFIG
    ids_path = BASE_DIR / "the_verdict_train_ids_small.txt"
    with open(ids_path, "r") as f:
        token_ids = [int(line) for line in f if line.strip()]

    ctx = cfg['context_length']
    inputs, targets = [], []
    # stride=ctx（窗口不重叠），与 C++ DataLoader 对齐。
    for i in range(0, len(token_ids) - ctx, ctx):
        inputs.append(token_ids[i:i + ctx])
        targets.append(token_ids[i + 1:i + ctx + 1])
    dataset = TensorDataset(torch.tensor(inputs), torch.tensor(targets))
    train_loader = DataLoader(dataset, batch_size=2, shuffle=False, drop_last=True)

    model = GPTModel(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4, weight_decay=0.1)

    num_epochs = 30
    global_step = -1
    last_loss = 0.0
    model.train()
    for epoch in range(num_epochs):
        for input_batch, target_batch in train_loader:
            input_batch, target_batch = input_batch.to(device), target_batch.to(device)
            optimizer.zero_grad()
            logits = model(input_batch)
            loss = torch.nn.functional.cross_entropy(
                logits.flatten(0, 1), target_batch.flatten(0, 1)
            )
            loss.backward()
            optimizer.step()
            global_step += 1
            last_loss = loss.item()
            print(f"[step {global_step}] epoch={epoch} train_loss={last_loss:.5f}")
    print(f"main small final train_loss: {last_loss:.5f}")


def run_normal():
    """完整配置：跑完整训练并绘制 loss 曲线。"""
    import matplotlib.pyplot as plt
    import tiktoken
    from train.train import train_model_simple

    cfg = GPT_CONFIG_124M
    train_ratio = 0.90
    split_idx = int(train_ratio * len(text_data))
    train_data = text_data[:split_idx]
    val_data = text_data[split_idx:]

    train_loader = create_dataloader_v1(
        train_data,
        batch_size=2,
        max_length=cfg['context_length'],
        stride=cfg['context_length'],
        drop_last=True,
        shuffle=True,
        num_workers=0,
    )
    val_loader = create_dataloader_v1(
        val_data,
        batch_size=2,
        max_length=cfg['context_length'],
        stride=cfg['context_length'],
        drop_last=False,
        shuffle=False,
        num_workers=0,
    )

    model = GPTModel(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4, weight_decay=0.1)
    num_epochs = 10
    tokenizer = tiktoken.get_encoding("gpt2")
    train_losses, val_losses, tokens_seen = train_model_simple(
        model, train_loader, val_loader, optimizer, device, num_epochs,
        eval_freq=5, eval_iter=1, start_context="Every effort moves you",
        tokenizer=tokenizer
    )

    def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
        fig, ax1 = plt.subplots(figsize=(5, 3))
        ax1.plot(epochs_seen, train_losses, label="Training loss")
        ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
        ax1.set_xlabel("Epochs")
        ax1.set_ylabel("Loss")
        ax1.legend(loc="upper right")
        ax2 = ax1.twiny()  # A
        ax2.plot(tokens_seen, train_losses, alpha=0)  # B
        ax2.set_xlabel("Tokens seen")
        fig.tight_layout()
        plt.show()

    epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
    plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)


if is_small:
    run_small()
else:
    run_normal()
