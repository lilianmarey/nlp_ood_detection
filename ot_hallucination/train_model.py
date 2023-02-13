
#%%
import fairseq
from tokenizer import Tokenizer
from modified_model import ModifiedTransformer
import ot
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

#%%
tokenizer = Tokenizer()

#%%
model = fairseq.models.transformer.TransformerModel.from_pretrained('checkpoint',
    checkpoint_file='checkpoint_best.pt',
    data_name_or_path = 'data/wmt18_de-en',
)
model.eval()
# %%
input = "Ohne Frühstück machten wir uns wieder auf den Weg."
tokenized_input = tokenizer.encode(input)

#%%
encoded_output = model.translate(tokenized_input)
output = tokenizer._decode(encoded_output)
print("\n".join((input, output)))
#%%
model.models[0].encoder
# %%
model.models[0].decoder.layers[-1].fc1.
# %%
m = ModifiedTransformer()
# %%
m.attention_model.translate(tokenized_input)
# %%
# %%
a = model.generate(model.encode(tokenized_input), beam=6)
# %%
m.attention_model.encode(tokenized_input)

# %%
len(a)
# %%
a[0]
# %%
a[1]
# %%
a[0]['attention'].shape
# %%
distrib = a[0]['attention'].mean(axis=1).numpy()
u = ot.unif(len(distrib))
x = np.arange(len(distrib), dtype=np.float64).reshape(11,1)
# %%
x
#%%
M = ot.dist(x,x, metric='euclidean')
M /= M.max()*0.1
# %%
M
#%%
ot.emd2(u, distrib, M)
# %%
import matplotlib.pyplot as plt
# %%
plt.scatter(x, distrib)
plt.scatter(x, u)
plt.plot()
# %%
df = pd.read_csv("data/annotated_corpus.csv", index_col=0)
# %%
df[df['full-unsupport']==1]
# %%
(df['full-unsupport'] & ~df['repetitions'] & ~df['strong-unsupport'] & ~df['omission']).sum()
# %%
df.iloc[:,-5:].sum(axis=1).max()
# %%
len(df)-625
# %%
df['oscillatory-hall'] = df['repetitions']
# %%
df['strongly-detached-hall'] = df['strong-unsupport'] & ~df['oscillatory-hall']
# %%
df['fully_detached'] = df['full-unsupport']

# %%
df['label'] = ''
# %%
df['label'] = 'strongly-detached-hall' if df['strongly-detached-hall'] else df['label'] 
# %%
def labelize(line):
    if line[-6:-1].sum()>1:
        print(line)
    if line['repetitions']==1:
        return 'oscillatory-hall'
    elif line['strong-unsupport']==1 and line['omission']==0 and line['repetitions']==0 :
        if line['full-unsupport']==1:
            print('oy')
        return 'strongly-detached-hall'
    elif line['full-unsupport']==1:
        return 'fully_detached'
    elif line['named-entities']==1 or line['omission']==1 or line['mt']!=line['ref']:
        return 'incorrect'
    else : return 'correct'


#%%

df['label'] = df.apply(labelize, axis=1)
# %%
(df['label']=='strongly-detached-hall').sum()
# %%
df.groupby('label').count()
# %%
df[df.label=='uncorrect'][df.mt!=df.ref]
# %%
(df.iloc[:,-6:-1].sum(axis=0)>=1).count()
# %%
df.iloc[:,-6:-1].sum(axis=0).sum()
# %%
df[df.label=='uncorrect'][df.mt!=df.ref]
#%%
scores = {"oscillatory-hall":[],"strongly-detached-hall":[], "fully_detached":[], "correct":[]  }
progress = tqdm(total=len(df[df.label!='uncorrect']))
for cat in scores.keys():
    for _, line in df[df.label == cat].iterrows():
        tokenized_input = tokenizer.encode(line['src'])
        output = model.generate(model.encode(tokenized_input), beam=5)
        attn = output[0]['attention'].mean(axis=1).numpy()
        u = ot.unif(len(attn))
        score = np.sum(np.abs(attn - u))*.5
        scores[cat].append(score)
        progress.update()


    

# %%
import pickle
# %%
with open("scores.pkl", 'wb') as f :
    pickle.dump(scores,f)
# %%
hallucination = scores['oscillatory-hall']+scores["strongly-detached-hall"]+scores["fully_detached"]
correct = scores['correct']
# %%
import seaborn as sns
import matplotlib.pyplot as plt
# %%
sns.set_style('white')
# %%
sns.histplot(hallucination, label='hallucination', kde=True, stat='probability', bins=30, color='red')
sns.histplot(correct, label='correct', kde=True, stat='probability', bins=30)
plt.legend()
plt.title("Wass-to-unif")
plt.plot()
# %%
