import pandas as pd
import numpy as np
from scipy.spatial.distance import mahalanobis, cdist

from .aiirw import AI_IRW


def process_latent_df(df):
    """Process full latent dataframe to use the 
    classic mean aggregation
    """
    X = np.array(df.T)
    softmax_scores = X[:, -1]
    preds = X[:, -2]
    X = X[:, :-2]
    X = X.reshape((X.shape[0], 768, 12))
    X = np.mean(X, axis = 2)
    return X, softmax_scores, preds

def MahaDist(X, distrib):
    """Computes Mahanolobis distance between each 
    vectors of X (in or ood) and distrib vectors (train)
    """
    m = np.mean(distrib, axis=0).reshape(1, 768)
    VI= np.linalg.inv(np.cov(distrib.T))

    return cdist(XA = X, XB = m, metric = 'mahalanobis', VI=VI)

def error_threshold(in_distances, out_distances, threshold):
    """Computing Err score for a certain threshold
    """
    err = len([i for i in in_distances if i > threshold]) + len([i for i in out_distances if i <= threshold])
    return err / (len(in_distances) + len(out_distances))

def minimize_error(in_distances, out_distances, threshold_range):
    """Computes Err score from list of in_distances and out_distances (choosing best threshold)
    """
    errors = [error_threshold(in_distances, out_distances, threshold) for threshold in (threshold_range)]
    return np.min(errors)

def exp_aggregation_function(alpha):
    """Define alpha_exponential aggregation function from alpha
    """
    def f(df):
        X = np.array(df.T)
        softmax_scores = X[:, -1]
        preds = X[:, -2]
        X = X[:, :-2]
        X = X.reshape((X.shape[0], 768, 12)) 

        for i in range(12):
            X[:, :, i] = np.exp(alpha * i *  X[:, :, i])

        X = np.mean(X, axis = 2)
        return X, softmax_scores, preds

    return f

def pipeline_exponential_aggregation(df_in, df_out, df, aggregation_funtion):
    """Building Err score for a certain alpha 
    with alpha_exponential aggregation function
    """
    # Aggregating with alpha_exponential aggregation
    X_out, softmax_out, preds_out = aggregation_funtion(df_out)
    X_in, softmax_in, preds_in = aggregation_funtion(df_in)
    X, softmax, preds = aggregation_funtion(df)

    # Computing Maha. similarities
    maha_in = np.concatenate([
                            MahaDist(X_in[preds_in == 0], X[preds == 0]), 
                            MahaDist(X_in[preds_in == 1], X[preds == 1])
                            ])
    maha_out = np.concatenate([
                            MahaDist(X_out[preds_out == 0], X[preds == 0]), 
                            MahaDist(X_out[preds_out == 1], X[preds == 1])
                            ])
    # Computing IRW similarities
    IRW_in = np.concatenate([
                            AI_IRW(X=X[preds == 0], AI=True, X_test=X_in[preds_in == 0], n_dirs=1000), 
                            AI_IRW(X=X[preds == 1], AI=True, X_test=X_in[preds_in == 1], n_dirs=1000)
                            ])
    IRW_out = np.concatenate([
                            AI_IRW(X=X[preds == 0], AI=True, X_test=X_out[preds_out == 0], n_dirs=1000), 
                            AI_IRW(X=X[preds == 1], AI=True, X_test=X_out[preds_out == 1], n_dirs=1000)
                            ])

    # Computing siilarities dataframes
    sim_df = pd.DataFrame(np.concatenate([maha_in, maha_out]), columns=['Maha'])
    sim_df['origin'] = ['in'] * len(maha_in) + ['out'] * len(maha_out)
    sim_df['IRW'] = -np.concatenate([IRW_in, IRW_out])

    # Computing Err scores
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