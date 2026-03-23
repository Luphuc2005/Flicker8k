import torch
import torch.nn as nn
import torchvision.models as models
class PositionalEmbedding(nn.Module):
    def __init__(self, vocab_size,embed_dim,max_len=25):
        super().__init__()
        self.token_embed=nn.Embedding(vocab_size, embed_dim) # tạo từ điển vector, vocab_size là số lượng thực thể, embed_dim là độ sâu kiến thức về thực thể đó.
        self.pos_embed=nn.Embedding(max_len, embed_dim) # tạo ra 5x5 vị trí có dimen =512
    def forward(self,x):
        B,T=x.shape# b là bachsize, t là số max-len=25
        if T > self.pos_embed.num_embeddings:
            old_embed = self.pos_embed
            new_embed = nn.Embedding(T, old_embed.embedding_dim).to(x.device)
            with torch.no_grad():
                new_embed.weight[:old_embed.num_embeddings] = old_embed.weight
                nn.init.normal_(
                    new_embed.weight[old_embed.num_embeddings:], mean=0.0, std=0.02
                )
            self.pos_embed = new_embed
        positions=torch.arange(0, T, device=x.device, dtype=torch.long).unsqueeze(0)
        token=self.token_embed(x)
        pos=self.pos_embed(positions)
        return token+pos