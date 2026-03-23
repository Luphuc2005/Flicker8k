import torch 
import torch.nn as nn
import torchvision.models as models
class CNNEncoder(nn.Module):
    def __init__(self, embed_dim=512):
        super().__init__()
        efficientnet=models.efficientnet_b0(pretrained=True)
        self.backbone=efficientnet.features # bo classfier
        self.conv=nn.Conv2d(1280, embed_dim, kernel_size=1)
    def forward(self,images):
        features=self.backbone(images) #(B, 1280,7,7)
        features=self.conv(features) #(B, emb_dim(512), 7,7) -> giảm chiều dữ liệu, không dùng dense
        features=features.flatten(2) #(B, 512,49) trải phẳng
        features=features.permute(0,2,1)#(B,49,512) đổi chỗ
        return features

        