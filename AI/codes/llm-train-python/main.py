import sys
from pathlib import Path

import torch

from model.GPTModel import GPTModel

# 两档配置：
#   normal：GPT-2 124M 完整配置（12 层），跑完整训练并绘制 loss 曲线。
#   small ：小配置（3 层），逐 step 打印 loss，用于快速冒烟和与 C++ 版逐 step 对齐。
# 默认自动选择设备并跑 normal。
# 兼容旧用法：`python main.py small`。
# 新用法：`python main.py <cpu|cuda|metal|auto> <small|normal> [aligned|demo] [max_steps] [log_interval]`。
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

def select_device(device_arg):
    """Select a PyTorch device.

    `metal` is the user-facing name; PyTorch exposes it as `mps`.
    """
    name = (device_arg or "auto").lower()
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested CUDA device, but torch.cuda.is_available() is False")
        return torch.device("cuda")
    if name in ("metal", "mps"):
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise RuntimeError("Requested Metal/MPS device, but torch.backends.mps.is_available() is False")
        return torch.device("mps")
    raise RuntimeError(f"Unknown device: {device_arg}")


def parse_args(argv):
    profiles = {"small", "normal"}
    devices = {"cpu", "cuda", "metal", "mps", "auto"}
    modes = {"aligned", "demo"}
    profile_arg = "normal"
    device_arg = "auto"
    mode_arg = "aligned"
    max_steps_arg = 0
    log_interval_arg = 1
    numeric_args = []
    for arg in argv:
        value = arg.lower()
        if value in profiles:
            profile_arg = value
        elif value in devices:
            device_arg = value
        elif value in modes:
            mode_arg = value
        elif value.isdigit():
            numeric_args.append(int(value))
        else:
            raise RuntimeError(
                "Usage: python main.py [cpu|cuda|metal|auto] [small|normal] [aligned|demo] [max_steps] [log_interval]"
            )
    if len(numeric_args) > 0:
        max_steps_arg = numeric_args[0]
    if len(numeric_args) > 1:
        log_interval_arg = numeric_args[1]
    return device_arg, profile_arg, mode_arg, max_steps_arg, log_interval_arg


device_arg, profile, mode, max_steps, log_interval = parse_args(sys.argv[1:])
is_small = (profile == 'small')

torch.manual_seed(123)
device = select_device(device_arg)
print(f"llm-train-python device={device} profile={profile} mode={mode}")

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


def maybe_synchronize():
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def run_aligned(profile_name):
    """与 C++ train_gpt 对齐的纯训练模式。

    使用 tiktoken 导出的 token id，不 shuffle，drop_rate=0，drop_last=True；
    不做 eval / sample / plot，便于和 C++ CPU/Metal/CUDA benchmark 对比。
    """
    from torch.utils.data import DataLoader, TensorDataset

    cfg = dict(SMALL_CONFIG if profile_name == "small" else GPT_CONFIG_124M)
    cfg['drop_rate'] = 0.0
    ids_name = "the_verdict_train_ids_small.txt" if profile_name == "small" else "the_verdict_train_ids.txt"
    ids_path = BASE_DIR / ids_name
    with open(ids_path, "r") as f:
        token_ids = [int(line) for line in f if line.strip()]

    ctx = cfg['context_length']
    inputs, targets = [], []
    for i in range(0, len(token_ids) - ctx, ctx):
        inputs.append(token_ids[i:i + ctx])
        targets.append(token_ids[i + 1:i + ctx + 1])
    dataset = TensorDataset(torch.tensor(inputs), torch.tensor(targets))
    train_loader = DataLoader(dataset, batch_size=2, shuffle=False, drop_last=True)

    model = GPTModel(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4, weight_decay=0.1)
    num_epochs = 30 if profile_name == "small" else 10
    global_step = -1
    last_loss = None
    stop = False
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
            last_loss = loss
            should_log = log_interval > 0 and (global_step % log_interval == 0)
            if should_log:
                maybe_synchronize()
                print(f"[step {global_step}] epoch={epoch} train_loss={loss.item():.5f}")
            if max_steps > 0 and global_step + 1 >= max_steps:
                stop = True
                break
        if stop:
            break

    maybe_synchronize()
    final_loss = last_loss.item() if last_loss is not None else float("nan")
    print(f"main {device} {profile_name} aligned final train_loss: {final_loss:.5f}")


def run_normal():
    """完整配置：跑完整训练并绘制 loss 曲线。"""
    import matplotlib.pyplot as plt
    import tiktoken
    from tool.data_loader import create_dataloader_v1
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


if mode == "aligned":
    run_aligned(profile)
elif is_small:
    run_small()
else:
    run_normal()
