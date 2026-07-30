from tool.deps import require_module


def sample_next_token(logits, greedy, temperature):
    """根据最后一步 logits 选择下一个 token，支持贪心和温度采样。"""
    torch = require_module(
        "torch",
        "pip install -r AI/codes/llm-inference-python/requirements.txt",
    )
    if greedy or temperature <= 0:
        return torch.argmax(logits, dim=-1, keepdim=True)
    probs = torch.softmax(logits / temperature, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def generate(model, tokenizer, prompt, args, device):
    """执行逐 token 生成循环，并把生成 token 解码回文本。"""
    torch = require_module(
        "torch",
        "pip install -r AI/codes/llm-inference-python/requirements.txt",
    )
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    input_ids = encoded["input_ids"].to(device)
    generated = input_ids
    eos_id = tokenizer.eos_token_id
    with torch.no_grad():
        for _ in range(args.max_new_tokens):
            logits = model(generated)
            next_logits = logits[:, -1, :]
            next_id = sample_next_token(next_logits, args.greedy, args.temperature)
            generated = torch.cat([generated, next_id], dim=-1)
            if eos_id is not None and int(next_id.item()) == int(eos_id):
                break
    return tokenizer.decode(generated[0].tolist(), skip_special_tokens=True)
