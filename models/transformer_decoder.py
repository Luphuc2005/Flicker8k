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
    def forward(self,x,encoder_out):
        x=self.embedding(x) # co duoc vi tri
    #causal mask
        T=x.size(1)
        mask=torch.triu(torch.ones(T,T),diagonal=1).bool().to(x.device) #he đi (mask) những từ ở tương lai. Khi tính Attention, mô hình chỉ được "nhìn" về các từ đã viết phía trước.
        #self attention
        attn_out,_=self.self_att(x,x,x,attn_mask=mask) #
        x=self.norm1(x+attn_out)
#Mô hình xem xét mối quan hệ giữa các từ đã viết. Ví dụ: Nếu đã viết "The patient has...", từ tiếp theo khả năng cao liên quan đến bệnh lý.

        #cross attention
        attn_out,_=self.cross_att(x,encoder_out,encoder_out)
        x=self.norm2(x+attn_out)
#         Đây là lúc Decoder kết nối với Encoder:

# Query (Q): Đến từ x (những gì Decoder đang viết).

# Key (K) & Value (V): Đến từ encoder_out (thông tin hình ảnh từ Encoder).

# Tư duy: "Tôi đang viết đến đoạn mô tả phổi (Q), hãy cho tôi biết vùng phổi trong ảnh (K, V) có đặc điểm gì?".

        #feedforward
        ff_out=self.ff(x)
        x=self.norm3(x+ff_out)
        #Sau khi đã tổng hợp cả thông tin câu chữ và thông tin hình ảnh, dữ liệu đi qua lớp FeedForward để "chốt" lại kiến thức, sau đó lớp out sẽ dự đoán từ tiếp theo.
        return self.out(x)
    
