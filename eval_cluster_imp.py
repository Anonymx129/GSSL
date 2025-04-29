# Evaluation of the smooth transition from node level to cluster level  
# Generating metric results and plots  
# The parameters used in this script are optimal for the Cora dataset  
# Each dataset has its own optimal parameters  
# I run this script in VS Code with Jupyter support 

import warnings
warnings.filterwarnings('ignore')

import torch 
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid

cora_data = Planetoid(root='/tmp/Cora', name='Cora')
cora = cora_data[0]

import pandas as pd
import matplotlib.pyplot as plt 
from models.cluster_level_imp import train_and_extract_ID_LID_cluster_level_imp, link_prediction_evaluation_cluster_level_imp
from models.node_level import train_and_extract_ID_LID_node_level

nn_cc = train_and_extract_ID_LID_cluster_level_imp(data = cora, 
                                                   pretrained_weights = 'pretrained_models/cora/weights/node.pth', 
                                                   pretrained_model = 'pretrained_models/cora/full_model/node.pth', 
                                                   num_epochs=200)
 


nodelevel = train_and_extract_ID_LID_node_level(data = cora)

fig, axs = plt.subplots(1, 4, figsize=(15, 4))
fig.suptitle('Node To Cluster Improvement', fontsize=14)

y1 = list(pd.DataFrame(nodelevel)['accuracy']) + list(pd.DataFrame(nn_cc)['accuracy'])
axs[0].plot(y1)
axs[0].vlines(200, min(y1), max(y1), color = 'red')
axs[0].set_title('Accuracy')
axs[0].grid(True)

y2 = list(pd.DataFrame(nodelevel)['ID_global_mean']) + list(pd.DataFrame(nn_cc)['ID_global_mean'])
axs[1].plot(y2)
axs[1].vlines(200, min(y2), max(y2), color = 'red')
axs[1].set_title('Intrinsic dimension')
axs[1].grid(True)

y3 = list(pd.DataFrame(nodelevel)['LID_mean']) + list(pd.DataFrame(nn_cc)['LID_mean'])
axs[2].plot(y3)
axs[2].vlines(200, min(y3), max(y3), color = 'red')
axs[2].set_title('Linear intrinsic dimension')
axs[2].grid(True)

y3 = list(pd.DataFrame(nodelevel)['kmeans']) + list(pd.DataFrame(nn_cc)['kmeans'])
axs[3].plot(y3)
axs[3].vlines(200, min(y3), max(y3), color = 'red')
axs[3].set_title('Clustering accuracy kmeans')
axs[3].grid(True)
plt.tight_layout()
plt.show()


link_prediction_evaluation_cluster_level_imp(cora,    
                                        pretrained_weights = 'pretrained_models/cora/weights/node.pth', 
                                        pretrained_model = 'pretrained_models/cora/full_model/node.pth',        
                                        num_hidden = 128, 
                                        num_proj_hidden = 128, 
                                        num_layers = 2, 
                                        num_epochs = 200, 
                                        weight_decay = 0.00001)
