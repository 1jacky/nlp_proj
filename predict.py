import torch
from data_preprocess import load_json
from bert_multi_label_cls import BertMultiLabelCls
from transformers import BertTokenizer


label2idx_path = "./data/label2idx.json"
save_model_path ="./model/multi_label_cls.pth"

hidden_size =  768
label2idx = load_json(label2idx_path)
class_num = len(label2idx)
idx2lable ={idx:label for label,idx in label2idx.items() }
device ="cuda" if torch.cuda.is_available() else "cpu"
tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
max_len=142

model = BertMultiLabelCls(hidden_size=hidden_size,class_num=class_num)
model.load_state_dict(torch.load(save_model_path))
model.to(device)
model.eval()

def predict(texts):
    outputs = tokenizer(texts,return_tensors="pt",max_length=max_len,padding=True,truncation=True)
    logits = model(outputs["input_ids"].to(device),
                   outputs["attention_mask"].to(device),
                   outputs["token_type_ids"].to(device))
    logits = logits.cpu().tolist()
    print(logits)
    result=[]
    for sample in logits:
        pred_label=[]
        for idx, logit in enumerate(sample):
            if logit > 0.5:
                pred_label.append(idx2lable[idx])
        result.append(pred_label)
    return result


if __name__ == '__main__':
    texts = ["没有赠品，物流很快，服务态度好，值得购买。","还好，手环在手上带着蓝牙老是断开，断开之后又不会重连，","测量不是特别准，心率幅度太大，步数和华为手机差几千步。  其他的通知功能做的还不错。"]
    result = predict(texts)
    print(result)