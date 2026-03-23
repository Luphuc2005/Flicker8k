import torch
import torch.nn as nn
import torchvision.models as models
class CaptionModel(nn.Module):
    def __init__(self, encoder,transformer_encoder,decoder):
        super().__init__()
        self.encoder=encoder
        self.transformer_encoder=transformer_encoder
        self.decoder=decoder
    def forward(self,images,captions):
        features=self.encoder(images)
        features=self.transformer_encoder(features)
        outputs=self.decoder(captions,features)
        return outputs