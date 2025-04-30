import pandas as pd
import numpy as np
import torch
from dgl.data.utils import load_graphs
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import adjusted_rand_score
from sklearn import linear_model


def node_feats(dims):
    feats = np.random.uniform(-1, 1, size=(282, dims))
    feats = standardize_features(feats)
    feats = torch.tensor(feats).float()
    feats = feats.view(282, -1)
    return feats

def load_dataset():
    src_matrix = pd.read_pickle('../data/mob/src_matrix.pkl').to_numpy()
    dst_matrix = pd.read_pickle('../data/mob/dst_matrix.pkl').to_numpy()
    chk_hell = torch.load('../data/checkins/chk_hellinger.pt')
    geo_hell = torch.load('../data/landusage/geo_hellinger.pt')
    data = {}
    data['meta_paths'] = [['RR'],['RC','CR'], ['RTc', 'TcR'], ['RTo', 'ToR'], ['RTd', 'TdR']] #[['RR'],['RC','CR'], ['RTc', 'TcR'], ['RTo', 'ToR'], ['RTd', 'TdR']]
    # data['hidden']
    data['src_matrix'] = src_matrix
    data['dst_matrix'] = dst_matrix
    data['chk_hell'] = chk_hell
    data['geo_hell'] = geo_hell
    data['heterograph_unified'] = load_heteograph_unified()

    # For downstream applications
    income = pd.read_pickle("../data/task/per_captia_income.pkl").sort_values(['zone']).dropna()
    income['UJAE001'] = income['UJAE001'] / 1000
    data['income'] = income
    crowded = pd.read_pickle("../data/task/popularity.pkl").sort_values(['zone']).dropna()
    data['popularity'] = crowded
    poverty = pd.read_pickle("../data/task/poverty.pkl").sort_values(['zone']).dropna()
    data['poverty'] = poverty
    cd = pd.read_pickle('../data/task/cd.pkl')
    cd = cd.sort_values(['zone'])
    cd_labels = cd.boro_cd.values
    cd_labels = np.delete(cd_labels, [141], 0)  # delete central-park
    data['labels'] = cd_labels
    return data

def load_heteograph_unified():
    g = load_graphs("../data/HUG/hug.bin")
    return g

def standardize_features(feature):
    var = np.var(feature, axis=1, keepdims=True)
    mean = np.mean(feature, axis=1, keepdims=True)
    std_inv = np.power(var, -0.5)
    std_inv[np.isinf(std_inv)] = 0.
    feature = np.multiply((feature - mean), std_inv)
    feature = feature[np.newaxis]
    return feature

def evaluation_metrics(y_pred, y_test):
    y_pred[y_pred < 0] = 0
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mae, np.sqrt(mse), r2


def regression(X_train, Y_train, X_test):
    reg = linear_model.Ridge(alpha=1)
    reg.fit(X_train, Y_train)
    y_pred = reg.predict(X_test)
    return y_pred

def kf_regression(X, Y):
    kf = KFold(n_splits=5)
    y_preds = []
    y_truths = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        y_pred = regression(X_train, Y_train, X_test)
        y_preds.append(y_pred)
        y_truths.append(Y_test)
    return np.concatenate(y_preds), np.concatenate(y_truths)

def predict_regression(emb, target):
    emb = emb[target.zone.values]
    y_pred, y_test = kf_regression(emb, target.values[:, -1])
    mae, rmse, r2 = evaluation_metrics(y_pred, y_test)
    return mae, rmse, r2


def region_cluetering(emb, cd_labels):
    seed_list = [73, 34, 88, 17, 98, 32, 62, 33, 52, 48]  # for reproducibility
    n = 12
    nmi_list = []
    ars_list = []
    emb = np.delete(emb, [141],
                    0)  # remove central park (central park is not included in community districts)
    for seed in seed_list:
        kmeans = KMeans(n_clusters=n, random_state=seed)

        emb_labels = kmeans.fit_predict(emb)
        nmi_list.append(normalized_mutual_info_score(cd_labels, emb_labels))
        ars_list.append(adjusted_rand_score(cd_labels, emb_labels))
    nmi = np.array(nmi_list).mean()
    ars = np.array(ars_list).mean()
    return nmi, ars