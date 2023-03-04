import pandas as pd
import numpy as np
from tqdm import tqdm

from .utils._aggreg_utils import pipeline_exponential_aggregation, process_latent_df, exp_aggregation_function

# Importing precomputend embeddings dataframes
df = pd.read_csv('data/train_latent.csv')
df_in = pd.read_csv('data/in_latent.csv')
df_out = pd.read_csv('data/out_latent.csv')

# Computing baseline scores
scores = pipeline_exponential_aggregation(df_in, df_out, df, process_latent_df)

# Computing alpha_exp aggregation scores
results = []
for alpha in tqdm(np.linspace(.0001, .1, 1000)):
    exp_process = exp_aggregation_function(alpha) 
    scores_alpha = pipeline_exponential_aggregation(df_in, df_out, df, exp_process)
    results.append([alpha, scores_alpha[0] - scores[0], scores_alpha[1] - scores[1]])
    pd.DataFrame(results, columns=['alpha', 'maha', 'IRW']).to_csv('Temp_results.csv')

# Saving results
pd.DataFrame(results, columns=['alpha', 'maha', 'IRW']).to_csv('Err_results.csv')


