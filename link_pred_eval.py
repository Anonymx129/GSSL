# Link prediction evaluation of different abstraction levels   
# This script focuses on the Cora dataset  
# I run this script in VS Code with Jupyter support  
# The hyperparameters for running other datasets are provided in a YAML file

import warnings
warnings.filterwarnings('ignore')

from torch_geometric.datasets import Planetoid

cora_data = Planetoid(root='/tmp/Cora', name='Cora')
cora = cora_data[0]


from models.node_level import link_prediction_evaluation_node_level
from models.proximity_level import link_prediction_evaluation_proximity_level
from models.graph_level import link_prediction_evaluation_graph_level
from models.cluster_level import link_prediction_evaluation_cluster_level


link_prediction_evaluation_node_level(cora, 
                                      num_hidden = 256, 
                                      num_proj_hidden = 256, 
                                      num_layers = 2, 
                                      num_epochs = 200, 
                                      weight_decay = 0.00001)


link_prediction_evaluation_proximity_level(cora, 
                                      num_hidden = 256, 
                                      num_proj_hidden = 256, 
                                      num_layers = 2, 
                                      num_epochs = 200, 
                                      weight_decay = 0.00001)


link_prediction_evaluation_graph_level(cora, 
                                      num_hidden = 256, 
                                      num_proj_hidden = 256, 
                                      num_layers = 2, 
                                      num_epochs = 200, 
                                      weight_decay = 0.00001)

link_prediction_evaluation_cluster_level(cora, 
                                        num_hidden = 256, 
                                        num_proj_hidden = 256, 
                                        num_layers = 2, 
                                        num_epochs = 200, 
                                        weight_decay = 0.00001)