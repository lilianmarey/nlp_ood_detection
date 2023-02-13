from torch import nn
import torch
from copy import deepcopy
import fairseq


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class ModifiedTransformer(nn.Module):
    def __init__(self):
        super(ModifiedTransformer, self).__init__()
        self.attention_model = (
            fairseq.models.transformer.TransformerModel.from_pretrained(
                "checkpoint",
                checkpoint_file="checkpoint_best.pt",
                data_name_or_path="data/wmt18_de-en",
            )
        )

        self.attention_model.models[0].decoder.layers[-1].fc1 = Identity()
        self.attention_model.models[0].decoder.layers[-1].fc2 = Identity()
        self.attention_model.models[0].decoder.layers[-1].final_layer_norm = Identity()

    def forward(self, x):
        return self.attention_model(x)
