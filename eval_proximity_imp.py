# Evaluation of the smooth transition from node level to proximity level  
# Generating metric results and plots  
# The parameters used in this script are optimal for the Cora dataset  
# Each dataset has its own optimal parameters  
# I run this script in VS Code with Jupyter support  

import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid

cora_data = Planetoid(root='/tmp/Cora', name='Cora')
cora = cora_data[0]

import warnings
warnings.filterwarnings("ignore")


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle 

from models.node_level import train_and_extract_ID_LID_node_level
from models.proximity_level import train_and_extract_ID_LID_proximity_level
from models.proximity_level_imp import train_and_extract_ID_LID_proximity_level_imp

#### Models #### 
nn_cora_300 = train_and_extract_ID_LID_node_level(data = cora, num_epochs=300, title = "Node level embeddings")

nn_cora = train_and_extract_ID_LID_node_level(data = cora, title = "Node level embeddings")

nn_pp_cora = train_and_extract_ID_LID_proximity_level(data = cora, pretrained_weights = 'pretrained_models/cora/weights/node.pth', pretrained=True, num_epochs=300, title="Node to proximity embeddings before filtering")

nn_pp = train_and_extract_ID_LID_proximity_level(data = cora, pretrained=False, num_epochs=300, title = "Proximity level embeddings")

nn_pp_imp_cora = train_and_extract_ID_LID_proximity_level_imp(data = cora, pretrained_weights = 'pretrained_models/cora/weights/node.pth', num_epochs=300, disk=0.295, title="Node to proximity embeddings after filtering")

#### Plots ####

plt.title('Node to Proximity Classification Improvement', fontsize=14)
plt.plot(pd.DataFrame(nn_cora_300).accuracy[22:], label = 'Node level')
plt.plot(pd.DataFrame(nn_pp).accuracy[22:], label = 'Proximity level', color = 'black')
plt.plot(pd.DataFrame(nn_pp_cora).accuracy[22:], label = 'Node to Proximity without filtering', color='r')
plt.plot(pd.DataFrame(nn_pp_imp_cora).accuracy[22:], label = 'Node to Proximity using filtering', color='g')
plt.grid(True)
plt.tight_layout()
plt.legend()
plt.show()


fig, axs = plt.subplots(2, 2, figsize=(12, 7))
fig.suptitle("Node to Proximity Improvement", fontsize=18, fontweight='bold')
plt.subplots_adjust(top=1)

y0 = (pd.DataFrame(nn_cora).ID_global_mean.to_list() + pd.DataFrame(nn_pp_cora).ID_global_mean.to_list())
axs[0,0].plot(range(0, 200), pd.DataFrame(nn_cora).ID_global_mean.to_list(), label='Pre-training')
axs[0,0].plot(range(200, 500), pd.DataFrame(nn_pp_cora).ID_global_mean.to_list(), label='Fine tuning without filtering')
axs[0,0].plot(range(200, 500), pd.DataFrame(nn_pp_imp_cora).ID_global_mean.to_list(), label='Fine tuning with filtering')
axs[0,0].set_title("ID", fontsize=18)
axs[0,0].vlines(200, min(y0), max(y0), color = 'red')
axs[0,0].grid(True)

y1 = (pd.DataFrame(nn_cora).LID_mean.to_list() + pd.DataFrame(nn_pp_cora).LID_mean.to_list())
axs[0,1].plot(range(0, 200), pd.DataFrame(nn_cora).LID_mean.to_list())
axs[0,1].plot(range(200, 500), pd.DataFrame(nn_pp_cora).LID_mean.to_list())
axs[0,1].plot(range(200, 500), pd.DataFrame(nn_pp_imp_cora).LID_mean.to_list())
axs[0,1].set_title("LID", fontsize=18)
axs[0,1].vlines(200, min(y1), max(y1), color = 'red')
axs[0,1].grid(True)

y2 = (pd.DataFrame(nn_cora).accuracy.to_list() + pd.DataFrame(nn_pp_cora).accuracy.to_list())
axs[1,0].plot(range(0, 200), pd.DataFrame(nn_cora).accuracy.to_list())
axs[1,0].plot(range(200, 500), pd.DataFrame(nn_pp_cora).accuracy.to_list())
axs[1,0].plot(range(200, 500), pd.DataFrame(nn_pp_imp_cora).accuracy.to_list())
axs[1,0].set_title("Accuracy", fontsize=18)
axs[1,0].vlines(200, min(y2), max(y2), color = 'red')
axs[1,0].grid(True)

y3 = (pd.DataFrame(nn_cora).kmeans.to_list() + pd.DataFrame(nn_pp_cora).kmeans.to_list())
axs[1,1].plot(range(0, 200), pd.DataFrame(nn_cora).kmeans.to_list())
axs[1,1].plot(range(200, 500), pd.DataFrame(nn_pp_cora).kmeans.to_list())
axs[1,1].plot(range(200, 500), pd.DataFrame(nn_pp_imp_cora).kmeans.to_list())
axs[1,1].set_title("Clustering", fontsize=18)
axs[1,1].vlines(200, min(y3), max(y3), color = 'red')
axs[1,1].grid(True)

fig.legend(loc='upper center', bbox_to_anchor=(0.5, -0.02), ncol=3)
plt.tight_layout()
plt.show()