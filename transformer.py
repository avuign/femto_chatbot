import torch
import torch.nn as nn
from self_attention import Attention_Block


def GELU(x):
    return (
        0.5
        * x
        * (
            1
            + torch.tanh(
                torch.sqrt(torch.tensor(2.0 / torch.pi))
                * (x + 0.044715 * torch.pow(x, 3))
            )
        )
    )


class SimpleLayer(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super().__init__()
        self.layer1 = nn.Linear(embedding_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, embedding_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = GELU(x)
        x = self.layer2(x)
        return x


class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class Transformer_Block(nn.Module):
    def __init__(
        self, embedding_dim, hidden_dim, context_length, n_heads, drop_rate, qkv_bias
    ):
        super().__init__()
        self.att = Attention_Block(
            d_in=embedding_dim,
            d_out=embedding_dim,
            context_length=context_length,
            num_heads=n_heads,
            dropout=drop_rate,
            qkv_bias=qkv_bias,
        )
        self.layer = SimpleLayer(embedding_dim, hidden_dim)
        self.norm1 = LayerNorm(embedding_dim)
        self.norm2 = LayerNorm(embedding_dim)
        self.drop_shortcut = nn.Dropout(drop_rate)

    def forward(self, x):
        # Shortcut connection for attention block
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        # Shortcut connection for feed forward block
        shortcut = x
        x = self.norm2(x)
        x = self.layer(x)
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        return x
