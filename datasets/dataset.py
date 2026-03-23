import torch 
from torch.utils.data import Dataset
from PIL import Image
class FlickrDataset(Dataset):
    def __init__(self, image_paths, captions, vocab, transform=None):
        self.image_paths=image_paths
        self.captions=captions
        self.vocab=vocab # đổi tượng lớp vocab đã tạo
        self.transform=transform#các phép biến đổi ảnh
        #-> nhiệm vụ là lưu trữ thông tin cần thiết
    def __len__(self):
        return len(self.image_paths)
    #-> cho pytỏch biết bộ dữ liệu này có tổng cộng bao nhiêu cặp (ảnh <-> caption). Khi train thfi pytỏch sẽ dựa vào con số này để biết khi nào thf hết 1 vòng

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")

        if self.transform:
            image = self.transform(image)

        caption = self.captions[idx]

        tokens = self.vocab.numericalize(caption)

        numericalized = [self.vocab.stoi["<start>"]]
        numericalized += tokens
        numericalized.append(self.vocab.stoi["<end>"])

        return image, torch.tensor(numericalized)