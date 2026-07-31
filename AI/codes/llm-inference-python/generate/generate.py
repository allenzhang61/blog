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
    if input_ids.shape[1] == 0:
        raise RuntimeError(
            f"输入编码后为空（tokenizer 未能编码该 prompt）：{prompt!r}"
        )
    use_kv_cache = getattr(args, "use_kv_cache", True)
    eos_id = tokenizer.eos_token_id
    with torch.no_grad():
        if use_kv_cache:
            generated = _generate_with_cache(
                model, input_ids, args, eos_id, torch
            )
        else:
            generated = _generate_no_cache(
                model, input_ids, args, eos_id, torch
            )
    return tokenizer.decode(generated[0].tolist(), skip_special_tokens=True)


def _generate_with_cache(model, input_ids, args, eos_id, torch):
    """增量解码：prefill 一次全量前向建立 KV 缓存，之后每步只前向 1 个新 token。"""
    generated = input_ids
    # prefill：喂入完整 prompt，得到缓存和最后一个位置的 logits
    logits, past = model(input_ids, use_cache=True)
    next_id = sample_next_token(logits[:, -1, :], args.greedy, args.temperature)
    generated = torch.cat([generated, next_id], dim=-1)
    if eos_id is not None and int(next_id.item()) == int(eos_id):
        return generated
    for _ in range(args.max_new_tokens - 1):
        logits, past = model(next_id, past_key_values=past, use_cache=True)
        next_id = sample_next_token(logits[:, -1, :], args.greedy, args.temperature)
        generated = torch.cat([generated, next_id], dim=-1)
        if eos_id is not None and int(next_id.item()) == int(eos_id):
            break
    return generated


def _generate_no_cache(model, input_ids, args, eos_id, torch):
    """无缓存基线：每步都把已生成的完整序列重新前向。"""
    generated = input_ids
    for _ in range(args.max_new_tokens):
        logits = model(generated)
        next_id = sample_next_token(logits[:, -1, :], args.greedy, args.temperature)
        generated = torch.cat([generated, next_id], dim=-1)
        if eos_id is not None and int(next_id.item()) == int(eos_id):
            break
    return generated
