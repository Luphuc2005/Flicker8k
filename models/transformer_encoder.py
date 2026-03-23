import torch
import torch.nn as nn
import torchvision.models as models
class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim=512, num_heads=1):
        super().__init__()
        self.norm=nn.LayerNorm(embed_dim)
        self.attn=nn.MultiheadAttention(embed_dim,num_heads,batch_first=True)
        self.ff=nn.Linear(embed_dim,embed_dim)
    def forward(self,x):
        x=self.norm(x)
        att_out,_=self.attn(x,x,x)
        x=x+att_out
        x=self.norm(x)
        x=self.ff(x)
        return x