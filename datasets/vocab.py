#cầu nối text <-> số
#model không hiểu "a dog is running" -> nó hiểu "[45,12,9,233]"
#itos: index to string, stoi: string to index
import collections
class Vocabulary:
    def __init__(self,freq_threshold=5):
        self.freq_threshold=freq_threshold #-> giữ từ phổ bién ->  model học tốt hơn
        self.itos={
            0:"<pad>", # ví batch cần dùng lenght
            1:"<start>", # cho decoder biết bắt đàu sinh từ đâu
            2:"<end>", # cho biết khi nào dừng
            3:"<unk>" # nếu không có
        }
        self.stoi={v: k for k,v in self.itos.items()}
    def __len__(self):
        return len(self.itos)
    def build_vocab(self, sentence_list):
        frequincies=collections.Counter()
        idx=4 # tạo index kế tiếp, không lặp lại 4 tk trên kia
        for sentence in sentence_list:
            for word in sentence.split():
                frequincies[word]+=1
                if frequincies[word]==self.freq_threshold: 
                    self.stoi[word]=idx
                    self.itos[idx]=word
                    idx+=1
    def numericalize(self,text):
        tokenized=text.split()
        return [
            self.stoi.get(token,self.stoi["<unk>"])
            for token in tokenized
        ]
