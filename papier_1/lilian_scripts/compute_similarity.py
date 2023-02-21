import pandas as pd
import numpy as np

df_train = pd.read_csv('train_latent.csv')
df_in = pd.read_csv('in_latent.csv')
df_out = pd.read_csv('out_latent.csv')


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

def D_M(x, class_0_variables, class_1_variables):
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



def save_distances(df, file_name):
    distances = []
    X = np.array(df)
    for i in range(X.shape[1]):
        v = X[:, 0]
        softmax = v[-1]
        pred = v[-2]
        v = np.reshape(v[:-2], (768, 12))

        avg_v = np.mean(v, axis = 1)

        distances.append([D_MSP(softmax), D_energy(), -D_M(), D_IRW()])
    
    pd.dataFrame(np.array(distances), columns = ['MSP', 'E', 'D_M', 'D_IRW']).to_csv('in_distances.csv')



