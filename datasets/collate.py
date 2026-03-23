import torch
from torch.nn.utils.rnn import pad_sequence


def collate_fn(batch):
    images = []
    captions = []

    for img, cap in batch:
        images.append(img)
        captions.append(cap)

    images = torch.stack(images)
    captions = pad_sequence(
        captions,
        batch_first=True,
        padding_value=0  # <pad>
    )

    return images, captions