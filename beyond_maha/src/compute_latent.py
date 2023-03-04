# Building dataframes containing the complete latent 
# representation of datasets in the BERT model
# A latent vector contains the output of 
# all BERT layers + softmax score + prediction 
# in a vector of size 12 * 768 + 2

from transformers import AutoModelForSequenceClassification, AutoTokenizer
import numpy as np
from datasets import load_dataset

from .utils._latent_utils import compute_all_latent
from .utils._hooks import features_hooks


# Import pretrained BERT finetuned on IMDB Database
tokenizer = AutoTokenizer.from_pretrained("fabriceyhc/bert-base-uncased-imdb")
model = AutoModelForSequenceClassification.from_pretrained(
                                "fabriceyhc/bert-base-uncased-imdb"
                                                            )

# Add hooks to model
feat_hook = [model.base_model.encoder.layer[i].register_forward_hook(features_hooks[i]) for i in range(12)]

################### PARAMETERS ###################

# train dataset
dataset = load_dataset("imdb")
inputs = [dataset['train'][i]['text'] for i in range(len(dataset['train']))]
df = compute_all_latent(inputs)
df.to_csv('train_latent.csv', index=False)

# in distribution
dataset = load_dataset("imdb")
inputs = [i['text'] for i in np.random.choice(dataset['test'], 2000)]
df = compute_all_latent(inputs)
df.to_csv('in_latent.csv', index=False)

# ood
dataset = load_dataset("ss")
inputs = [i['sentence'] for i in np.random.choice(dataset['test'], 2000)]
df = compute_all_latent(inputs)
df.to_csv('out_latent.csv', index=False)
