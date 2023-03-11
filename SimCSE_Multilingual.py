from transformers import AutoModel, AutoTokenizer
import torch
from transformers import AutoTokenizer,AutoModelForMaskedLM
model =AutoModelForMaskedLM.from_pretrained('IDEA-CCNL/Erlangshen-SimCSE-110M-Chinese')
tokenizer = AutoTokenizer.from_pretrained('IDEA-CCNL/Erlangshen-SimCSE-110M-Chinese')

import torch
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn.functional as F

texta = '操作简单，表盘选项也很多，总之比起以前买的颂拓，功能真的很强大,价格很贵，性价比不怎么样'
textb = '功能强大'
textc = '性价比高'
inputs_a = tokenizer(texta,return_tensors="pt")
inputs_b = tokenizer(textb,return_tensors="pt")
inputs_c = tokenizer(textc,return_tensors="pt")

outputs_a = model(**inputs_a ,output_hidden_states=True)
texta_embedding = outputs_a.hidden_states[-1][:,0,:].squeeze()

outputs_b = model(**inputs_b ,output_hidden_states=True)
textb_embedding = outputs_b.hidden_states[-1][:,0,:].squeeze()

outputs_c = model(**inputs_c ,output_hidden_states=True)
textc_embedding = outputs_c.hidden_states[-1][:,0,:].squeeze()
# if you use cuda, the text_embedding should be textb_embedding.cpu().numpy()
# 或者用torch.no_grad():

# with torch.no_grad():
#     silimarity_soce = cosine_similarity(texta_embedding.cpu().numpy().reshape(1,-1),textb_embedding.cpu().numpy().reshape(1,-1))[0][0]
# print(silimarity_soce)

print("texta: ",texta,"textb: ",textb,' 的相似度 ',F.cosine_similarity(texta_embedding,textb_embedding,dim=0))
print("texta: ",texta,"textc: ",textc,' 的相似度 ',F.cosine_similarity(texta_embedding,textc_embedding,dim=0))
