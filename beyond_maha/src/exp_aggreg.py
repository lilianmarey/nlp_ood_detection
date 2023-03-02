print("begining")

import pandas as pd
import numpy as np
from scipy.spatial.distance import mahalanobis, cdist

import plotly.express as px
import pandas as pd

from aiirw import AI_IRW

from tqdm import tqdm

print("importing data frames")

df = pd.read_csv('data/train_latent.csv')
df_in = pd.read_csv('data/in_latent.csv')
df_out = pd.read_csv('data/out_latent.csv')

def MahaDist(X, distrib):
    m = np.mean(distrib, axis=0).reshape(1, 768)
    VI= np.linalg.inv(np.cov(distrib.T))
    return cdist(XA = X, XB = m, metric = 'mahalanobis', VI=VI)

def error_threshold(in_distances, out_distances, threshold):
    err = len([i for i in in_distances if i > threshold]) + len([i for i in out_distances if i <= threshold])
    return err / (len(in_distances) + len(out_distances))

def minimize_error(in_distances, out_distances, threshold_range):
    errors = [error_threshold(in_distances, out_distances, threshold) for threshold in (threshold_range)]
    return np.min(errors)

def exp_aggregation_function(p):
    def f(df):
        X = np.array(df.T)
        softmax_scores = X[:, -1]
        preds = X[:, -2]
        X = X[:, :-2]
        X = X.reshape((X.shape[0], 768, 12)) 

        for i in range(12):
            X[:, :, i] = np.exp(p * i *  X[:, :, i])

        X = np.mean(X, axis = 2)
        return X, softmax_scores, preds

    return f

def pipeline_alternative_aggregation(df_in, df_out, df, aggregation_funtion):
    X_out, softmax_out, preds_out = aggregation_funtion(df_out)
    X_in, softmax_in, preds_in = aggregation_funtion(df_in)
    X, softmax, preds = aggregation_funtion(df)

    maha_in = np.concatenate([MahaDist(X_in[preds_in == 0], X[preds == 0]), MahaDist(X_in[preds_in == 1], X[preds == 1])])
    maha_out = np.concatenate([MahaDist(X_out[preds_out == 0], X[preds == 0]), MahaDist(X_out[preds_out == 1], X[preds == 1])])

    IRW_in = np.concatenate([AI_IRW(X=X[preds == 0], AI=True, X_test=X_in[preds_in == 0], n_dirs=1000), AI_IRW(X=X[preds == 1], AI=True, X_test=X_in[preds_in == 1], n_dirs=1000)])
    IRW_out = np.concatenate([AI_IRW(X=X[preds == 0], AI=True, X_test=X_out[preds_out == 0], n_dirs=1000), AI_IRW(X=X[preds == 1], AI=True, X_test=X_out[preds_out == 1], n_dirs=1000)])

    sim_df = pd.DataFrame(np.concatenate([maha_in, maha_out]), columns=['Maha'])
    sim_df['origin'] = ['in'] * len(maha_in) + ['out'] * len(maha_out)
    sim_df['IRW'] = -np.concatenate([IRW_in, IRW_out])


    score_maha = minimize_error(
                    (sim_df[sim_df['origin'] == 'in']['Maha']),
                    (sim_df[sim_df['origin'] == 'out']['Maha']),
                    list(sim_df[sim_df['origin'] == 'in']['Maha']) + list(sim_df[sim_df['origin'] == 'in']['Maha'])  

                            )


    score_IRW = minimize_error(
                    (sim_df[sim_df['origin'] == 'in']['IRW']),
                    (sim_df[sim_df['origin'] == 'out']['IRW']),
                    list(sim_df[sim_df['origin'] == 'in']['IRW']) + list(sim_df[sim_df['origin'] == 'in']['IRW'])  
                         )                     

    return (score_maha, score_IRW)

def process_latent_df(df):
    X = np.array(df.T)
    softmax_scores = X[:, -1]
    preds = X[:, -2]
    X = X[:, :-2]
    X = X.reshape((X.shape[0], 768, 12))
    X = np.mean(X, axis = 2)
    return X, softmax_scores, preds

print("computing baseline scores")
scores = pipeline_alternative_aggregation(df_in, df_out, df, process_latent_df)
print(scores)

power_mean_results = []

for p in tqdm(np.linspace(.0001, 1, 500)):
    exp_process = exp_aggregation_function(p) 
    scores_p = pipeline_alternative_aggregation(df_in, df_out, df, exp_process)
    power_mean_results.append([p, scores_p[0] - scores[0], scores_p[1] - scores[1]])
    pd.DataFrame(power_mean_results, columns = ['p', 'maha', 'IRW']).to_csv('Temp_.csv')

