import torch
import torch.nn as nn
import torchvision.models as models
class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

        self.ff = nn.Sequential(
            nn.Linear(embed_dim, 2048),
            nn.ReLU(),
            nn.Linear(2048, embed_dim)
        )

    def forward(self, x):
        # Attention block
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]

        # Feedforward block
        x = x + self.ff(self.norm2(x))

        return x