import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import sys
from pathlib import Path

# Thêm đường dẫn root để import các module
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.cnn_encoder import CNNEncoder
from models.transformer_encoder import TransformerEncoder
from models.transformer_decoder import TransformerDecoder
from models.caption_model import CaptionModel
from datasets.vocab import Vocabulary
from datasets.preprocessing import load_captions, filter_valid_images, flatten_data

def greedy_decode(model, image, vocab, max_len=20, device="cpu", return_attention=False):
    """
    Sinh caption bằng phương pháp Greedy Search.
    return_attention: Nếu True, trả về kèm trọng số attention để làm XAI.
    """
    model.eval()
    with torch.no_grad():
        features = model.encoder(image)
        features = model.transformer_encoder(features)

        start_token = vocab.stoi["<start>"]
        captions = [start_token]
        all_attention_weights = []
        
        for _ in range(max_len):
            caption_tensor = torch.LongTensor(captions).unsqueeze(0).to(device)
            
            if return_attention:
                output, attn_weights = model.decoder(caption_tensor, features, return_attention=True)
                # attn_weights shape: (batch, n_heads, query_len, key_len)
                # Lấy attention của từ cuối cùng (query index -1) ứng với 49 vùng ảnh
                all_attention_weights.append(attn_weights[0, :, -1, :].cpu())
            else:
                output = model.decoder(caption_tensor, features)
            
            last_word_logits = output[0, -1, :]
            predicted_id = last_word_logits.argmax().item()
            
            captions.append(predicted_id)
            if predicted_id == vocab.stoi["<end>"]:
                break
                
    words = [vocab.itos[idx] for idx in captions if idx not in [vocab.stoi["<start>"], vocab.stoi["<end>"], vocab.stoi["<pad>"]]]
    
    if return_attention:
        return " ".join(words), torch.stack(all_attention_weights)
    return " ".join(words)

def beam_search_decode(model, image, vocab, beam_size=3, max_len=20, device="cpu"):
    """
    Sinh caption bằng thuật toán Beam Search.
    """
    model.eval()
    with torch.no_grad():
        features = model.encoder(image)
        features = model.transformer_encoder(features)

        start_token = vocab.stoi["<start>"]
        beams = [([start_token], 0.0)]
        
        for _ in range(max_len):
            all_candidates = []
            for seq, score in beams:
                if seq[-1] == vocab.stoi["<end>"]:
                    all_candidates.append((seq, score))
                    continue
                
                caption_tensor = torch.LongTensor(seq).unsqueeze(0).to(device)
                output = model.decoder(caption_tensor, features)
                
                log_probs = torch.log_softmax(output[0, -1, :], dim=-1)
                top_k_log_probs, top_k_ids = log_probs.topk(beam_size)
                
                for i in range(beam_size):
                    next_seq = seq + [top_k_ids[i].item()]
                    next_score = score + top_k_log_probs[i].item()
                    all_candidates.append((next_seq, next_score))
            
            all_candidates.sort(key=lambda x: x[1], reverse=True)
            beams = all_candidates[:beam_size]
            
            if all(seq[-1] == vocab.stoi["<end>"] for seq, _ in beams):
                break
                
        best_seq, _ = beams[0]
        
    words = [vocab.itos[idx] for idx in best_seq if idx not in [vocab.stoi["<start>"], vocab.stoi["<end>"], vocab.stoi["<pad>"]]]
    return " ".join(words)

def load_model_and_vocab(checkpoint_path, data_dir, device):
    from datasets.data_loader import get_loaders
    print("Mô hình đang tải Vocabulary chuẩn...")
    _, vocab, _ = get_loaders(data_dir)
    
    print(f"Đang tải trọng số từ: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    checkpoint_config = checkpoint.get("config", {})
    max_len = checkpoint_config.get("max_len", 42)
    embed_dim = checkpoint_config.get("embed_dim", 512)
    num_heads = checkpoint_config.get("num_head", 2)
    ff_dim = checkpoint_config.get("ff_dim", 512)
    
    encoder = CNNEncoder(embed_dim=embed_dim)
    trans_enc = TransformerEncoder(embed_dim=embed_dim, num_heads=num_heads)
    decoder = TransformerDecoder(
        vocab_size=len(vocab),
        embed_dim=embed_dim,
        num_head=num_heads,
        ff_dim=ff_dim,
        max_len=max_len
    )
    model = CaptionModel(encoder, trans_enc, decoder).to(device)
    
    if 'model_state' in checkpoint:
        model.load_state_dict(checkpoint['model_state'])
    elif 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    return model, vocab

def main():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Thay đổi đường dẫn thực tế trên máy em
    from config import config
    CHECKPOINT = os.path.join(config.CHECKPOINT_DIR, "epoch_9.pt")
    DATA_DIR = config.DATA_DIR
    IMAGE_PATH = os.path.join(DATA_DIR, "Images", "1000268201_693b08cb0e.jpg")
    
    if not os.path.exists(CHECKPOINT):
        print(f"LỖI: Không tìm thấy checkpoint tại {CHECKPOINT}")
        return

    model, vocab = load_model_and_vocab(CHECKPOINT, DATA_DIR, DEVICE)
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    img = Image.open(IMAGE_PATH).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)
    
    print("\n--- GREEDY SEARCH ---")
    print(greedy_decode(model, img_tensor, vocab, device=DEVICE))
    
    print("\n--- BEAM SEARCH (k=3) ---")
    print(beam_search_decode(model, img_tensor, vocab, beam_size=3, device=DEVICE))

if __name__ == "__main__":
    main()
