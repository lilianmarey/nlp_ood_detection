
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

# %%
def labelize(line):
    if line['repetitions']==1:
        return 'oscillatory-hall'
    elif line['strong-unsupport']==1 and line['omission']==0 and line['repetitions']==0 :
        if line['full-unsupport']==1:
            print('oy')
        return 'strongly-detached-hall'
    elif line['full-unsupport']==1:
        return 'fully_detached'
    elif line['named-entities']==1 or line['omission']==1 or line['mt'].lower()!=line['ref'].lower():
        return 'incorrect'
    else : return 'correct'


#%%
df = pd.read_csv('data/annotated_corpus.csv', index_col=0)
df['label'] = df.apply(labelize, axis=1)
#%%
df.groupby('label').count()
# %%
#%%
scores = {"oscillatory-hall":[],"strongly-detached-hall":[], "fully_detached":[], "correct":[],'incorrect':[]   }
progress = tqdm(total=len(df))
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
r_held = scores['correct']+scores['incorrect']
# %%
import seaborn as sns
import matplotlib.pyplot as plt
# %%
sns.set_style('white')
# %%
sns.histplot(hallucination, label='Hallucination', kde=True, stat='probability', color='red', bins=30, line_kws={"lw":3})
sns.histplot(r_held, label="Heldout R_held", kde=True, stat='probability', bins=30, line_kws={"lw":3})
plt.legend()
plt.title("Wass-to-unif")
plt.plot()
# %%
