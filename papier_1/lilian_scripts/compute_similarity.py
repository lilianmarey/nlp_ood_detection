import pandas as pd
import numpy as np
from tqdm import tqdm
from aiirw import AI_IRW

df_train = pd.read_csv('data/train_latent.csv').T

df_train_0 = df_train[df_train[9216] == 0][range(12*768)]
df_train_1 = df_train[df_train[9216] == 1][range(12*768)]

def compute_class_variables(df):

    mean_vectors = []
    for i, row in tqdm(df.iterrows()):
        mean_vector = np.mean(np.reshape(np.array(row), (768, 12)), axis = 1)
        mean_vectors.append(mean_vector)

    X = np.array(mean_vectors)
    cov_matrix = np.cov(X.T)
    precision_matrix = np.linalg.inv(cov_matrix)
    esperance_vector = np.mean(X, axis = 0)
    class_variables = (esperance_vector, cov_matrix)

    return class_variables, X

class_0_variables, vectors_0 = compute_class_variables(df_train_0)
class_1_variables, vectors_1 = compute_class_variables(df_train_1)

df_train_0 = None
df_train_1 = None


def MSP(x):
    """Max softmax proba
    x is the max logits value
    """
    return 1 - x

def D_energy(x, T = 1):
    """
    x is the max logits value
    """
    return T * np.log(np.exp(x / T) + np.exp((1 - x) / T))

def D_M(x, prediction, class_0_variables, class_1_variables):
    """Mahalanobis distance
    """
    if prediction == 0:
        esperance = class_0_variables[0]
        precision = class_0_variables[1]
    elif prediction == 1:
        esperance = class_1_variables[0]
        precision = class_1_variables[1]
    else:
        return ValueError

    v = x - esperance
    u = 1 + (v.T @ precision @ v)
    return 1 / u

def D_IRW(x, prediction, vectors_0, vectors_1):
    if prediction == 0:
        train_vectors = vectors_0
    elif prediction == 1:
        train_vectors = vectors_1
    else:
        return ValueError
    
    return AI_IRW(X=train_vectors, AI=True, X_test=np.reshape(x, (1, 768)), n_dirs=1000)[0]

def save_similarities(df, file_name):
    distances = []
    X = np.array(df)
    for i in tqdm(range(X.shape[0])):
        v = X[i, :]
        softmax = v[-1]
        pred = v[-2]
        v = np.reshape(v[:-2], (768, 12))

        avg_v = np.mean(v, axis = 1)

        distances.append([
                        MSP(softmax), 
                        D_energy(softmax), 
                        -D_M(avg_v, pred, class_0_variables, class_1_variables), 
                        D_IRW(avg_v, pred, vectors_0, vectors_1)
                            ])
    
    pd.DataFrame(np.array(distances), columns = ['MSP', 'E', 'D_M', 'D_IRW']).to_csv(file_name)


df_in = pd.read_csv('data/out_latent.csv').T
# df_out = pd.read_csv('data/out_latent.csv')

save_similarities(df_in, 'out_distances.csv')