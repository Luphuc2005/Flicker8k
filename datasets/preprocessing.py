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
            img,cap=line.split(",",1)#chia dòng thành 2 phần, dựa trên dấu phẩy. img=tên ảnh, cap là ndung
            cap=cap.lower().strip()
            #strip(): Hàm này dùng để loại bỏ tất cả các ký tự khoảng trắng (bao gồm dấu cách, tab, và ký tự xuống dòng \n) ở hai đầu (đầu và cuối) của một chuỗi. Nó không tác động đến khoảng trắng ở giữa chuỗi.
            cap="<start> " + cap +" <end>"
            if img not in captions_dict:
                captions_dict[img]=[] #nếu ảnh đã xuất hiện -> tạo 1 ds trống cho nó
            captions_dict[img].append(cap)  # thêm câu mô tả vào ds của ảnh tương ứng
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