#%%
import fairseq
from tokenizer import Tokenizer
import numpy as np
from tqdm import tqdm
import torch
import pickle

#%%
model = fairseq.models.transformer.TransformerModel.from_pretrained('checkpoint',
    checkpoint_file='checkpoint_best.pt',
    data_name_or_path = 'data/wmt18_de-en',
    bpe='sentencepiece', 
    sentencepiece_model = 'checkpoint/sentencepiece.joint.bpe.model'
)
model.eval()
tokenizer = Tokenizer()
# %%
def get_dict(lang='en'):
    dico = {}
    with open(f"data/wmt18_de-en/dict.{lang}.txt", 'r') as f:
        lines = f.readlines()
        dico = [el.split() for el in lines ]
        dico = {k:v for k,v in dico}
    return dico



#%%
src_dict = get_dict('de')
tgt_dict = get_dict()
# %%
src_dict
# %%
ds = fairseq.data.LanguagePairDataset("data/wmt18_de-en/valid", src_dict=src_dict, tgt_dict=tgt_dict, src_sizes=27)

# %%
src_dict = fairseq.data.Dictionary()
tgt_dict = fairseq.data.Dictionary()
# %%
src_dict.add_from_file("data/wmt18_de-en/dict.de.txt")
tgt_dict.add_from_file("data/wmt18_de-en/dict.en.txt")

# %%
for a in ds:
    print(a)
# %%

class fairseq_pre_dataset(torch.utils.data.Dataset):
    def __init__(self):
        super(self ).__init__()
        with open('data/wmt18_de-en'):
        data = 

#%%

model.task.load_dataset('valid')
# %%
data_bin = model.task.datasets['valid']
# %%
train_pairs = [
    (a := model.decode(item['source']), model.translate(a), model.decode(item['target'])) 
    for item in data_bin
]
# %%
train_pairs[1]
# %%
with open("pairs.pkl", 'wb') as f :
    pickle.dump(train_pairs,f)
# %%
from comet import download_model, load_from_checkpoint

# %%
model_path = download_model("wmt20-comet-da")
model = load_from_checkpoint(model_path)
# %%
pairs = []
with open(r"pairs.pkl", 'rb') as f :
    pairs=pickle.load(f)

#%%
pairs
#%%
train_pairs
# %%
pairs = [{"src":a,"mt":b, "ref":c } for a, b, c in pairs ]
# %%
model_output = model.predict(pairs, batch_size=8, gpus=1)

# %%
seg_scores, system_score = model_output.scores, model_output.system_score

# %%
model_output[0]
# %%
