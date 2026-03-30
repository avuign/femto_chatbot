import torch
import torch.nn as nn
from transformer import LayerNorm, Transformer_Block


# Model
class Femto_GPT(nn.Module):
    def __init__(
        self,
        voc_size,
        n_layers,
        embedding_dim,
        hidden_dim,
        context_length,
        n_heads,
        drop_rate,
        qkv_bias,
    ):
        super().__init__()
        self.tok_emb = nn.Embedding(voc_size, embedding_dim)
        self.pos_emb = nn.Embedding(context_length, embedding_dim)
        self.drop_emb = nn.Dropout(drop_rate)

        self.trf_blocks = nn.Sequential(
            *[
                Transformer_Block(
                    embedding_dim,
                    hidden_dim,
                    context_length,
                    n_heads,
                    drop_rate,
                    qkv_bias,
                )
                for _ in range(n_layers)
            ]
        )

        self.final_norm = LayerNorm(embedding_dim)
        self.out_head = nn.Linear(embedding_dim, voc_size, bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits
