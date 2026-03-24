import os
from PIL import Image

#Đọc file .txt hoặc .csv và gom tất cả các mô tả (captions) của cùng một ảnh vào một nhóm.
def load_captions(captions_file):
    captions_dict={} #key là ảnh, value là dsach các câu mô tả của từng ảnh
    with open(captions_file, "r",encoding="utf-8") as f: #mở file
        for line in f:    #duyệt từng file
            line=line.strip()
            if len(line)==0:#bỏ qua các dòng trống trong file txt
                continue

            # Skip common CSV header lines.
            lowered = line.lower()
            if lowered in {"image,caption", "image_name,comment_number,comment"}:
                continue

            # Support both formats:
            # 1) captions.txt: image.jpg,caption text
            # 2) Flickr8k.token.txt: image.jpg#0\tcaption text
            if "\t" in line:
                left, cap = line.split("\t", 1)
                img = left.split("#", 1)[0]
            elif "," in line:
                img, cap = line.split(",", 1)
            else:
                continue

            img = img.strip()
            cap = cap.lower().strip()
            if img not in captions_dict:
                captions_dict[img]=[] #nếu ảnh đã xuất hiện -> tạo 1 ds trống cho nó
            captions_dict[img].append(cap)  # thêm câu mô tả vào ds của ảnh tương ứng

    if not captions_dict:
        raise ValueError(f"Khong doc duoc caption nao tu file: {captions_file}")

    return captions_dict

#"Dọn rác". Kiểm tra xem file ảnh có thực sự tồn tại và có bị hỏng hay không trước khi huấn luyện.
def filter_valid_images(image_dir, captions_dict):
    valid={} # tạo từ điển chỉ chứa ảnh sạch
    for img in captions_dict:
        path=os.path.join(image_dir, img) #nối giữa thhư mục với tên ảnh để có đường dẫn
        try:
            Image.open(path).convert("RGB")
            valid[img]=captions_dict[img] #nếu mở ảnh thành công, lấy đường dẫn của ảnh 
        except:
            continue
    return valid
#"Phẳng hóa" dữ liệu. Chuyển từ dạng từ điển phức tạp sang 2 danh sách song song để nạp vào mô hình PyTorch dễ dàng hơn.
def flatten_data(image_dir, captions_dict):
    image_paths, captions = [], []
    for img,caps in captions_dict.items():
        for cap in caps:
            image_paths.append(os.path.join(image_dir, img))
            captions.append(cap)
    return image_paths, captions