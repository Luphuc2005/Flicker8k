import torch
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
import os
from evaluation.metrics import calculate_metrics

@torch.no_grad()
def greedy_decode(model, image, vocab, max_len=20, device="cpu"):
    """
    Sinh caption bằng phương pháp Greedy Search (tối giản cho evaluation).
    """
    model.eval()
    features = model.encoder(image)
    features = model.transformer_encoder(features)

    start_token = vocab.stoi["<start>"]
    captions = [start_token]
    
    for _ in range(max_len):
        caption_tensor = torch.LongTensor(captions).unsqueeze(0).to(device)
        output = model.decoder(caption_tensor, features)
        
        last_word_logits = output[0, -1, :]
        predicted_id = last_word_logits.argmax().item()
        
        captions.append(predicted_id)
        if predicted_id == vocab.stoi["<end>"]:
            break
            
    words = [vocab.itos[idx] for idx in captions if idx not in [vocab.stoi["<start>"], vocab.stoi["<end>"], vocab.stoi["<pad>"]]]
    return " ".join(words)

def evaluate_model(model, vocab, device, image_paths, captions_dict, transform=None, limit=None):
    """
    Đi qua danh sách ảnh, sinh caption và tính toán metric.
    image_paths: Danh sách UNIQUE image paths.
    captions_dict: {image_name: [list of raw captions]}
    """
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    model.eval()
    gts = {}
    res = {}

    # Nếu có limit thì lấy một phần tập val để đánh giá nhanh
    paths_to_eval = image_paths[:limit] if limit else image_paths

    print(f"Evaluating metrics on {len(paths_to_eval)} images...")
    
    for i, path in enumerate(tqdm(paths_to_eval)):
        img_name = os.path.basename(path)
        img_id = i # Dùng index làm ID cho pycocoevalcap

        # Prepare Image
        try:
            img = Image.open(path).convert("RGB")
            img_tensor = transform(img).unsqueeze(0).to(device)
        except Exception as e:
            print(f"Skip {path}: {e}")
            continue

        # Generate Caption
        prediction = greedy_decode(model, img_tensor, vocab, device=device)

        # Store for pycocoevalcap
        res[img_id] = [{"caption": prediction}]
        gts[img_id] = [{"caption": c} for c in captions_dict[img_name]]

    # Calculate Score
    if not res:
        return {}
        
    scores = calculate_metrics(gts, res)
    return scores
