import torch
import torch.nn as nn
import torchvision.models as models
from models.positional_embedding import PositionalEmbedding
class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size,embed_dim=512,num_head=2, ff_dim=512, max_len=25):
        super().__init__()
        self.embedding=PositionalEmbedding(vocab_size,embed_dim,max_len) # giúp decoder biết từ nào đang đứng vị trí nào trongc câu trả lời
        self.self_att=nn.MultiheadAttention(embed_dim,num_head,batch_first=True) # giúp các từ trong câu dang tạo ra tự liên kết với nhau
        self.cross_att=nn.MultiheadAttention(embed_dim,num_head,batch_first=True) #Nó giúp Decoder "hỏi" Encoder: "Với hình ảnh bạn đã phân tích, tôi nên viết từ tiếp theo là gì?".
        self.norm1=nn.LayerNorm(embed_dim)
        self.norm2=nn.LayerNorm(embed_dim)
        self.norm3=nn.LayerNorm(embed_dim)
        self.ff=nn.Sequential(
            nn.Linear(embed_dim,ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim,embed_dim)
        )
        self.out=nn.Linear(embed_dim,vocab_size)
        #: Lớp cuối cùng để biến các vector 512 chiều thành xác suất của các từ trong từ điển (ví dụ: xác suất từ "Cat" là 0.8, "Dog" là 0.1).
    def forward(self, x, encoder_out, return_attention=False):
        x = self.embedding(x)
        # causal mask
        T = x.size(1)
        mask = torch.triu(torch.ones(T, T), diagonal=1).bool().to(x.device)
        # self attention
        attn_out, _ = self.self_att(x, x, x, attn_mask=mask)
        x = self.norm1(x + attn_out)

        # cross attention
        attn_out, attn_weights = self.cross_att(x, encoder_out, encoder_out, average_attn_weights=False)
        x = self.norm2(x + attn_out)

        # feedforward
        ff_out = self.ff(x)
        x = self.norm3(x + ff_out)
        
        logits = self.out(x)
        if return_attention:
            return logits, attn_weights
        return logits
