from transformers import AutoModelForSequenceClassification, AutoTokenizer, TextClassificationPipeline
import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
from datasets import load_dataset


# Import pretrained BERT finetuned on IMDB Database

tokenizer = AutoTokenizer.from_pretrained("fabriceyhc/bert-base-uncased-imdb")

model = AutoModelForSequenceClassification.from_pretrained("fabriceyhc/bert-base-uncased-imdb")

# Hook functions to get latent representation of data

def features_hook_0(model, inp, output):
    global feat_0
    feat_0 = output

def features_hook_1(model, inp, output):
    global feat_1
    feat_1 = output

def features_hook_2(model, inp, output):
    global feat_2
    feat_2 = output

def features_hook_3(model, inp, output):
    global feat_3
    feat_3 = output

def features_hook_4(model, inp, output):
    global feat_4
    feat_4 = output

def features_hook_5(model, inp, output):
    global feat_5
    feat_5 = output

def features_hook_6(model, inp, output):
    global feat_6
    feat_6 = output

def features_hook_7(model, inp, output):
    global feat_7
    feat_7 = output

def features_hook_8(model, inp, output):
    global feat_8
    feat_8 = output

def features_hook_9(model, inp, output):
    global feat_9
    feat_9 = output

def features_hook_10(model, inp, output):
    global feat_10
    feat_10 = output

def features_hook_11(model, inp, output):
    global feat_11
    feat_11 = output

features_hooks = [features_hook_0, features_hook_1, features_hook_2, 
                    features_hook_3, features_hook_4, features_hook_5, 
                    features_hook_6, features_hook_7, features_hook_8, 
                    features_hook_9, features_hook_10, features_hook_11]



feat_hook = [model.base_model.encoder.layer[i].register_forward_hook(features_hooks[i]) for i in range(12)]

def label_to_bin(label):
    if label == 'neg':
        return 0
    return 1

def get_latent(input_data):
    pipe = TextClassificationPipeline(
                                    model=model, tokenizer=tokenizer
                                    )
    pred = pipe(input_data)[0]
    feats = [feat_0[0], feat_1[0], feat_2[0], 
                    feat_3[0], feat_4[0], feat_5[0], 
                    feat_6[0], feat_7[0], feat_8[0], 
                    feat_9[0], feat_10[0], feat_11[0]]
    label = label_to_bin(pred['label'])
    softmax = pred['score']

    pred_vector = torch.tensor([label, softmax])
    latent = np.concatenate([torch.cat([feats[i][0, 0] for i in range(12)], dim = 0).numpy(), pred_vector.numpy()])
    return latent

def compute_all_latent(inputs):
    latent_list = []
    for x in tqdm(inputs):
        try:
            latent_list.append(get_latent(x))
        except:
            pass
    df = pd.DataFrame(latent_list)
    df.columns = list(range(768 * 12)) + ['pred', 'softmax_score']
    return df.T

################### PARAMETERS ###################

dataset = load_dataset("imdb")

# inputs = [dataset['train'][i]['text'] for i in range(len(dataset['train']))]
inputs = [i['text'] for i in np.random.choice(dataset['test'], 2000)]

df = compute_all_latent(inputs)
df.to_csv('in_latent.csv', index = False)