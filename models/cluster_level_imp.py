import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

import numpy as np
import random

from models.intrinsic_dimension import computeLID, computeID

from models.utils import *
from models.first_eval_protocol_func import *

import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


class Model(nn.Module):
    def __init__(self, encoder: Encoder, num_hidden: int, num_proj_hidden: int,
                 tau: float = 0.5, num_clust: int = 7, alpha: float = 1, 
                 cluster_centers = None):
        super(Model, self).__init__()
        
        self.encoder: Encoder = encoder
        self.tau = tau
        self.cluster_number = num_clust

        self.fc1 = torch.nn.Linear(num_hidden, num_proj_hidden)
        self.fc2 = torch.nn.Linear(num_proj_hidden, num_hidden)
        
        self.embedding_dimension = num_hidden
        self.alpha = alpha

        if cluster_centers is None:
            initial_cluster_centers = torch.zeros(self.cluster_number, self.embedding_dimension, dtype=torch.float)
            nn.init.xavier_uniform_(initial_cluster_centers)
        else:
            initial_cluster_centers = cluster_centers
        self.cluster_centers = nn.Parameter(initial_cluster_centers)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        enc = self.encoder(x, edge_index)
        return enc
    
    def projection(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)

    def soft_clustering(self, embeddings):
        p_squared = torch.sum((embeddings.unsqueeze(1) - self.cluster_centers)**2, 2)
        p = 1.0 / (1.0 + (p_squared / self.alpha))
        power = float(self.alpha + 1) / 2
        p = p ** power
        p_dist = (p.t() / torch.sum(p, 1)).t()
        return p_dist

    def predict(self, embeddings):
        q = self.soft_clustering(embeddings)
        out = np.argmax(q.detach().cpu().numpy(), axis=1)
        centroids = self.cluster_centers[out]
        return torch.tensor(centroids, dtype=torch.float)
    
    def loss(self, z1: torch.Tensor, z2: torch.Tensor, z3: torch.Tensor,
             mean: bool = True):
        z1 = self.projection(z1)
        z2 = self.projection(z2)
        z3 = self.projection(z3)
        
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        z3 = F.normalize(z3)

        c1 = self.predict(z1)
        c2 = self.predict(z2)
        c3 = self.predict(z3)

        f = lambda x1, x2 : torch.exp(torch.mm(x1, x2)/ self.tau)

        l1 =  -torch.log(f(z1,c2.T).diag() / (f(z1,c1.T).sum(1) - f(z1,c1.T).diag() + f(z1,c2.T).sum(1) - f(z1,c2.T).diag() + f(z1,c3.T).sum(1)))
        l2 =  -torch.log(f(z2,c1.T).diag() / (f(z2,c2.T).sum(1) - f(z2,c2.T).diag() + f(z2,c1.T).sum(1) - f(z2,c1.T).diag() + f(z2,c3.T).sum(1)))

        ret = (l1 + l2) * 0.5
        ret = ret.mean() if mean else ret.sum()

        return ret


def q_mat(X, centers, alpha=1.0):
    if X.size == 0:
        q = np.array([])
    else:
        q = 1.0 / (1.0 + (np.sum(np.square(np.expand_dims(X, 1) - centers), axis=2) / alpha))
        q = q**((alpha + 1.0) / 2.0)
        q = np.transpose(np.transpose(q) / np.sum(q, axis=1))
    return q
  
def generate_unconflicted_data_index(emb, centers_emb, beta1, beta2):
    unconf_indices = []
    conf_indices = []
    q = q_mat(emb, centers_emb, alpha=1.0)
    confidence1 = np.zeros((q.shape[0],))
    confidence2 = np.zeros((q.shape[0],))
    a = np.argsort(q, axis=1)
    for i in range(q.shape[0]):
        confidence1[i] = q[i,a[i,-1]]
        confidence2[i] = q[i,a[i,-2]]
        if (confidence1[i]) > beta1 and (confidence1[i] - confidence2[i]) > beta2:
            unconf_indices.append(i)
        else:
            conf_indices.append(i)
    unconf_indices = np.asarray(unconf_indices, dtype=int)
    conf_indices = np.asarray(conf_indices, dtype=int)
    return unconf_indices, conf_indices

def visualize(h, color, title):
    z = TSNE(n_components=2).fit_transform(h.detach().cpu())
    color = color.detach().cpu()
    
    plt.figure(figsize=(7,7))
    plt.xticks([])
    plt.yticks([])

    plt.scatter(z[:, 0], z[:, 1], s=45, c=color, cmap="plasma")
    plt.title(title, fontsize = 19, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.axis('off')
    plt.show()

def train_and_extract_ID_LID_cluster_level_imp(  
            data,
            pretrained_weights = None,
            pretrained_model = None,
            plot = True, 
            title = None,
            learning_rate = 0.0005, 
            num_hidden = 128, 
            num_proj_hidden = 128,
            activation = nn.PReLU(),
            base_model = GCNConv,
            num_layers = 2,
            drop_edge_rate_1 = 0.2,
            drop_edge_rate_2 = 0.4,
            drop_feature_rate_1 = 0.3,
            drop_feature_rate_2 = 0.4,
            drop_scheme = 'uniform',
            tau = 0.4,
            num_epochs = 200,
            num_epochs_eval = 500,
            num_classes = 7,
            weight_decay = 0.00001,
            beta1 = 0.25, 
            beta2 = 0.1,
            rd_seed = 28,
        ):
          
    torch.manual_seed(rd_seed)
    random.seed(rd_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)

    full_model = torch.load(pretrained_model)
    embedding = full_model(data.x, data.edge_index)
    kmeans = KMeans(n_clusters=num_classes, random_state=rd_seed).fit(embedding.detach().cpu())


    encoder = Encoder(data.num_features, num_hidden, activation, base_model=base_model, k=num_layers).to(device)
    model = Model(encoder, num_hidden, num_proj_hidden, tau, cluster_centers=torch.tensor(kmeans.cluster_centers_, dtype=torch.float, requires_grad=True)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)


    model.load_state_dict(torch.load(pretrained_weights), strict=False)   
    
    dict = {'ID_global_mean' : [],
            'LID_mean' : [],
            'accuracy' : [],
            'nmi': [],
            'kmeans': [],}

    previous_unconflicted = []

    epoch_stable = 0 

    for epoch in range(0, num_epochs):
        
        optimizer.zero_grad()

        edge_index_1 = drop_edge(data, device, drop_scheme, drop_edge_rate_1)
        edge_index_2 = drop_edge(data, device, drop_scheme, drop_edge_rate_2)
        
        x_1 = drop_feature_global(data, device, drop_scheme, drop_feature_rate_1)
        x_2 = drop_feature_global(data, device, drop_scheme, drop_feature_rate_2)
        x_3 = shuffle(data.x)
        
        z1 = model(x_1, edge_index_1)
        z2 = model(x_2, edge_index_2)
        z3 = model(x_3, data.edge_index)

        z = model(data.x, data.edge_index)
        ct = model.cluster_centers
        
        if epoch % 15 == 0 :
            unconflicted_ind, _ = generate_unconflicted_data_index(z.detach().cpu().numpy(), ct.detach().cpu().numpy(), beta1, beta2)
            print(len(unconflicted_ind))
        
        if len(previous_unconflicted) < len(unconflicted_ind) :
            z1_unconf = z1[unconflicted_ind]
            z2_unconf = z2[unconflicted_ind]
            z3_unconf = z3[unconflicted_ind]
            
            previous_unconflicted = unconflicted_ind
        else:
            epoch_stable += 1
            z1_unconf = z1[previous_unconflicted]
            z2_unconf = z2[previous_unconflicted]
            z3_unconf = z3[previous_unconflicted]
        if epoch_stable >= 10:
            epoch_stable = 0
            beta1 = beta1 * 0.95 
            beta2 = beta2 * 0.85
        
        loss = model.loss(z1_unconf, z2_unconf, z3_unconf)
        loss.backward()
        optimizer.step()

        print(f'(T) | Epoch={epoch:03d}, loss={loss.item():.4f}')

        res = classifier(model, data, LogReg(num_hidden, num_classes), device, n_epochs = num_epochs_eval)
        dict['accuracy'].append(res[0])

        res_clust = clustering_evaluation(model, data, num_classes)
        dict['kmeans'].append(res_clust[0])
        dict['nmi'].append(res_clust[1])

        ID = []
        LID = []

        z = model(data.x, data.edge_index)

        for k in range(num_classes):
            ID.append(computeID(z[data.y == k]))
            LID.append(computeLID(z[data.y == k]))
                
        ID = np.asarray(ID)
        LID = np.asarray(LID)
        ID_global_mean = np.mean(ID)
        LID_mean = np.mean(LID)
        dict['LID_mean'].append(LID_mean)
        dict['ID_global_mean'].append(ID_global_mean)     
    
    print("=== Final ===")

    if plot:
        visualize(z, data.y, title)
        
    return dict


from models.linkpred import *
from tqdm import tqdm

def link_prediction_evaluation_cluster_level_imp(  
            data,
            pretrained_weights = None,
            pretrained_model = None,
            learning_rate = 0.0005, 
            num_hidden = 128, 
            num_proj_hidden = 128,
            activation = nn.PReLU(),
            base_model = GCNConv,
            num_layers = 2,
            drop_edge_rate_1 = 0.2,
            drop_edge_rate_2 = 0.4,
            drop_feature_rate_1 = 0.3,
            drop_feature_rate_2 = 0.4,
            drop_scheme = 'uniform',
            tau = 0.4,
            num_epochs = 200,
            wait=200,
            weight_decay = 0.00001,
            beta1 = 0.25, 
            beta2 = 0.1,
            num_classes = 7,
            rd_seed = 129,
        ):
          
    torch.manual_seed(rd_seed)
    random.seed(rd_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)

    full_model = torch.load(pretrained_model)
    embedding = full_model(data.x, data.edge_index)
    kmeans = KMeans(n_clusters=num_classes, random_state=rd_seed).fit(embedding.detach().cpu())


    encoder = Encoder(data.num_features, num_hidden, activation, base_model=base_model, k=num_layers).to(device)
    model = Model(encoder, num_hidden, num_proj_hidden, tau, cluster_centers=torch.tensor(kmeans.cluster_centers_, dtype=torch.float, requires_grad=True)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)


    model.load_state_dict(torch.load(pretrained_weights), strict=False)     
    
    split_edge = do_edge_split_direct(data)
    data.edge_index = to_undirected(split_edge['train']['edge'].t())

    predictor = link_decoder

    best_valid = 0.0
    best_epoch = 0
    cnt_wait = 0
    best_result=0


    with tqdm(total=num_epochs, desc='(T)') as pbar:

        previous_unconflicted = []
        epoch_stable = 0
        
        for epoch in range(0, num_epochs):
            optimizer.zero_grad()

            edge_index_1 = drop_edge(data, device, drop_scheme, drop_edge_rate_1)
            edge_index_2 = drop_edge(data, device, drop_scheme, drop_edge_rate_2)

            x_1 = drop_feature_global(data, device, drop_scheme, drop_feature_rate_1)
            x_2 = drop_feature_global(data, device, drop_scheme, drop_feature_rate_2)
            x_3 = shuffle(data.x)

            z1 = model(x_1, edge_index_1)
            z2 = model(x_2, edge_index_2)
            z3 = model(x_3, data.edge_index)

            z = model(data.x, data.edge_index)
            ct = model.cluster_centers
            
            if epoch % 15 == 0 :
                unconflicted_ind, _ = generate_unconflicted_data_index(z.detach().cpu().numpy(), ct.detach().cpu().numpy(), beta1, beta2)
                print(len(unconflicted_ind))
            
            if len(previous_unconflicted) < len(unconflicted_ind) :
                z1_unconf = z1[unconflicted_ind]
                z2_unconf = z2[unconflicted_ind]
                z3_unconf = z3[unconflicted_ind]
                
                previous_unconflicted = unconflicted_ind
            else:
                epoch_stable += 1
                z1_unconf = z1[previous_unconflicted]
                z2_unconf = z2[previous_unconflicted]
                z3_unconf = z3[previous_unconflicted]
            if epoch_stable >= 10:
                epoch_stable = 0
                beta1 = beta1 * 0.95 
                beta2 = beta2 * 0.85
            
            loss = model.loss(z1_unconf, z2_unconf, z3_unconf)
            loss.backward()
            optimizer.step()

            pbar.set_postfix({'loss': loss})
            pbar.update()
            result = test_link_prediction(model, predictor, data, split_edge, num_hidden)
            valid_hits = result['auc_val']
            if valid_hits > best_valid:
                best_valid = valid_hits
                best_epoch = epoch
                best_result = result
                cnt_wait = 0
            else:
                cnt_wait += 1

            if cnt_wait == wait:
                print('Early stopping!')
                break
        test_auc = best_result['auc_test']

        print(f'Final result: Epoch:{best_epoch}, auc: {test_auc}' )

    return test_auc
