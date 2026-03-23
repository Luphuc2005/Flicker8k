import torch.nn as nn

def get_loss():
    return nn.CrossEntropyLoss(ignore_index=0)