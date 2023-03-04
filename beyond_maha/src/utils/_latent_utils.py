import pandas as pd
import numpy as np
import torch
from tqdm import tqdm

from transformers import TextClassificationPipeline


def label_to_bin(label):
    if label == 'neg':
        return 0
    return 1


def get_latent(input_data, model, tokenizer):
    """Get latent representation over all the Bert layer for 
    one data point + softmax score and prediction.
    feat_i ar global variables
    """
    pipe = TextClassificationPipeline(
                                        model=model, tokenizer=tokenizer
                                    )
    pred = pipe(input_data)[0]
    feats = [
            feat_0[0], feat_1[0], feat_2[0],
            feat_3[0], feat_4[0], feat_5[0],
            feat_6[0], feat_7[0], feat_8[0],
            feat_9[0], feat_10[0], feat_11[0]
            ]
    label = label_to_bin(pred['label'])
    softmax = pred['score']

    pred_vector = torch.tensor([label, softmax])
    latent = np.concatenate([
        torch.cat([feats[i][0, 0] for i in range(12)], dim=0).numpy(), pred_vector.numpy()
                            ])

    return latent


def compute_all_latent(inputs):
    """Build the latent representation + softmax score and 
    prediction on all input vectors
    """
    latent_list = []
    for x in tqdm(inputs):
        try:
            latent_list.append(get_latent(x))
        except:
            pass
    df = pd.DataFrame(latent_list)
    df.columns = list(range(768 * 12)) + ['pred', 'softmax_score']
    return df.T
