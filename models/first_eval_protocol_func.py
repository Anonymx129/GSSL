############################ EVAL ############################

import torch 
import torch.nn as nn
import torch.nn.functional as F

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, normalized_mutual_info_score, adjusted_rand_score

from models.logreg import LogReg
from munkres import Munkres
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd


torch.manual_seed(19)
torch.use_deterministic_algorithms(True)


def visualize(h, color):
    z = TSNE(n_components=2).fit_transform(h.detach().cpu().numpy())

    plt.figure(figsize=(10,10))
    plt.xticks([])
    plt.yticks([])

    plt.scatter(z[:, 0], z[:, 1], s=70, c=color, cmap="Set2")
    plt.show()


def train_lrc(X_train, y_train, model, optimizer, criterion, device):
    model.to(device).train()
    optimizer.zero_grad()  
    out = model(X_train)  
    loss = criterion(out, y_train)  
    loss.backward()  
    optimizer.step()  
    return loss

def test_lrc(X_test, y_test, model, device):
    model.to(device).eval()
    out = model(X_test)
    pred = out.argmax(dim=1)  
    return accuracy_score(pred.detach().cpu().numpy(), y_test.detach().cpu().numpy()), f1_score(pred.detach().cpu().numpy(), y_test.detach().cpu().numpy(),  average='macro')

def classifier(ssl_model, data, clf_model, device, n_epochs = 500):
    criterion = torch.nn.CrossEntropyLoss()  
    optimizer = torch.optim.Adam(clf_model.parameters(), lr=0.01, weight_decay=5e-4)  
    embedding = ssl_model(data.x, data.edge_index)

    X = embedding.detach()
    Y = data.y.detach()

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.9, random_state=28)
    
    for epoch in range(0, n_epochs):
        loss = train_lrc(X_train, y_train, clf_model, optimizer, criterion, device)

    return test_lrc(X_test, y_test, clf_model, device)

def get_matches(y_true, y_pred):
    l1 = list(set(y_true))
    numclass1 = len(l1)

    l2 = list(set(y_pred))
    numclass2 = len(l2)
       

    cost = np.zeros((numclass1, numclass2), dtype=int)
    for i, c1 in enumerate(l1):
        mps = [i1 for i1, e1 in enumerate(y_true) if e1 == c1]
        for j, c2 in enumerate(l2):
            mps_d = [i1 for i1 in mps if y_pred[i1] == c2]

            cost[i][j] = len(mps_d)

    m = Munkres()
    cost = cost.__neg__().tolist()

    indexes = m.compute(cost)

    new_predict = np.zeros(len(y_pred))
    for i, c in enumerate(l1):
        c2 = l2[indexes[i][1]]
        ai = [ind for ind, elm in enumerate(y_pred) if elm == c2]
        new_predict[ai] = c
    return new_predict

def clustering_evaluation(model, data, num_clusters):
    model.eval()
    z = model(data.x, data.edge_index)
    X = z.detach().cpu().numpy()
    Y = data.y.detach().cpu().numpy()

    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    pred_label1 = kmeans.fit_predict(X)

    new_predict1 = get_matches(Y, pred_label1)

    kmeans_accuracy = accuracy_score(Y, new_predict1)

    # ari = adjusted_rand_score(pred_label1, Y)
    nmi = normalized_mutual_info_score(pred_label1, Y)

    return kmeans_accuracy, nmi