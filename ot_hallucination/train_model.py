
#%%
import fairseq
from tokenizer import Tokenizer
from modified_model import ModifiedTransformer
import ot
import numpy as np

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
