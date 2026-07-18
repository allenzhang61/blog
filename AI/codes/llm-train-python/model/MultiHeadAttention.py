from torch import nn
import torch


class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        self.d_out = d_out
        self.num_heads = num_heads
        # 将投影维度缩小，以匹配期望的输出维度
        self.head_dim = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        # 使用线性层，组合头部输出
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)

        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        # 张量形状：b, num_tokens, d_out
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        # 将 d_out 拆成 num_heads * head_dim
        # 张量形状：(b, num_tokens, num_heads, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # 将张量的形状转置成 (b,num_heads,num_tokens,head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # (b,num_heads,num_tokens,head_dim) * (b,num_heads,head_dim, num_tokens)
        # = (b,num_heads,num_tokens,num_tokens)
        attn_scores = queries @ keys.transpose(2, 3)

        # 掩码被截断到 token 的数量
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        # 填充 -inf，让当前位置不能关注未来 token
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # (b,num_heads,num_tokens,num_tokens) * (b,num_heads,num_tokens,head_dim)
        # = (b,num_heads,num_tokens,head_dim)
        # ' = (b,num_tokens,num_heads,head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2)

        # 将后最两个维度再合并在一起：num_heads * head_dim = d_out
        # (b,num_tokens,d_out)
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)

        # (b, num_tokens, d_out)
        context_vec = self.out_proj(context_vec)

        return context_vec
