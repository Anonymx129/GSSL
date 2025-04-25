# Evaluation of different abstraction levels  
# Evaluation of the transitions between different abstraction levels  
# Generating metric results and plots  
# This script focuses on the Cora dataset  
# I run this script in VS Code with Jupyter support  
# The hyperparameters for running other datasets are provided in a YAML file

import warnings
warnings.filterwarnings('ignore')

from torch_geometric.datasets import Planetoid

cora_data = Planetoid(root='/tmp/Cora', name='Cora')
cora = cora_data[0]


from models.node_level import train_and_extract_ID_LID_node_level
from models.graph_level import train_and_extract_ID_LID_graph_level
from models.cluster_level import train_and_extract_ID_LID_cluster_level
from models.proximity_level import train_and_extract_ID_LID_proximity_level

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle 


#### Generate Results ####

nn = train_and_extract_ID_LID_node_level(data = cora, title='Node level 2D embeddings')
gg = train_and_extract_ID_LID_graph_level(data = cora, title='Graph level 2D embeddings')
cc = train_and_extract_ID_LID_cluster_level(data = cora, title='Cluster level 2D embeddings')
pp = train_and_extract_ID_LID_proximity_level(data = cora, title='Proximity level 2D embeddings')


gg_nn = train_and_extract_ID_LID_node_level(data = cora, pretrained_weights = 'pretrained_models/cora/weights/graph.pth', pretrained = True)
gg_pp = train_and_extract_ID_LID_proximity_level(data = cora, pretrained_weights = 'pretrained_models/cora/weights/graph.pth', pretrained = True)
gg_cc = train_and_extract_ID_LID_cluster_level(data = cora, pretrained_weights = 'pretrained_models/cora/weights/graph.pth', pretrained = True)

nn_gg = train_and_extract_ID_LID_graph_level(data = cora, pretrained_weights = 'pretrained_models/cora/weights/node.pth', pretrained = True)
nn_pp = train_and_extract_ID_LID_proximity_level(data = cora, pretrained_weights = 'pretrained_models/cora/weights/node.pth', pretrained = True)
nn_cc = train_and_extract_ID_LID_cluster_level(data = cora, pretrained_weights = 'pretrained_models/cora/weights/node.pth', pretrained = True)

pp_nn = train_and_extract_ID_LID_node_level(data = cora, pretrained_weights = 'pretrained_models/cora/weights/proximity.pth', pretrained = True)
pp_gg = train_and_extract_ID_LID_graph_level(data = cora, pretrained_weights = 'pretrained_models/cora/weights/proximity.pth', pretrained = True)
pp_cc = train_and_extract_ID_LID_cluster_level(data = cora, pretrained_weights = 'pretrained_models/cora/weights/proximity.pth', pretrained = True)

cc_nn = train_and_extract_ID_LID_node_level(data = cora, pretrained_weights = 'pretrained_models/cora/weights/cluster.pth', pretrained = True)
cc_gg = train_and_extract_ID_LID_graph_level(data = cora, pretrained_weights = 'pretrained_models/cora/weights/cluster.pth', pretrained = True)
cc_pp = train_and_extract_ID_LID_proximity_level(data = cora, pretrained_weights = 'pretrained_models/cora/weights/cluster.pth', pretrained = True)



res_node = {}
res_node['node'] = nn
res_node['graph'] = gg
res_node['cluster'] = cc
res_node['proximity'] = pp

# import json

# with open('res_node.json', 'w') as fp:
#     json.dump(res_node, fp)

# with open('res_node.json', 'r') as fp:
#     res_node = json.load(fp)

# res_node.keys()


dd = pd.concat([pd.DataFrame(nn), pd.DataFrame(pp), pd.DataFrame(cc), pd.DataFrame(gg),
            pd.DataFrame(gg_nn), pd.DataFrame(gg_pp), pd.DataFrame(gg_cc),
            pd.DataFrame(nn_gg), pd.DataFrame(nn_pp), pd.DataFrame(nn_cc), 
            pd.DataFrame(pp_nn), pd.DataFrame(pp_gg), pd.DataFrame(pp_cc), 
            pd.DataFrame(cc_nn), pd.DataFrame(cc_gg), pd.DataFrame(cc_pp),], axis = 0)

dd['transition'] = 'node'
dd.iloc[200:400, -1] = 'proximity'
dd.iloc[400:600, -1] = 'cluster'
dd.iloc[600:800, -1] = 'graph'
dd.iloc[800:1000, -1] = 'graph_to_node'
dd.iloc[1000:1200, -1] = 'graph_to_proximity'
dd.iloc[1200:1400, -1] = 'graph_to_cluster'
dd.iloc[1400:1600, -1] = 'node_to_graph'
dd.iloc[1600:1800, -1] = 'node_to_proximity'
dd.iloc[1800:2000, -1] = 'node_to_cluster'
dd.iloc[2000:2200, -1] = 'proximity_to_node'
dd.iloc[2200:2400, -1] = 'proximity_to_graph'
dd.iloc[2400:2600, -1] = 'proximity_to_cluster'
dd.iloc[2600:2800, -1] = 'cluster_to_node'
dd.iloc[2800:3000, -1] = 'cluster_to_graph'
dd.iloc[3000:3200, -1] = 'cluster_to_proximity'

dd.to_pickle('saved_results/cora first evaluation protocol/cora_ft_crr.pkl')
dd = pd.read_pickle('saved_results/cora first evaluation protocol/cora_ft_crr.pkl')


#### Plots ####
def plot_sum(dd, base, title, transitions, plot_titles):
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(14, 9))
    fig.suptitle(title, fontsize=14)

    row_titles = ['Accuracy', 'Intrinsic dimension', 'Linear intrinsic dimension']
    metrics = ['accuracy', 'ID_global_mean', 'LID_mean']

    for i, row in enumerate(axes):
        for j, ax in enumerate(row):
            y = (dd[dd['transition'] == base][metrics[i]].to_list() + dd[dd['transition'] == transitions[j]][metrics[i]].to_list())
            ax.vlines(200, min(y), max(y), color = 'red')
            ax.plot(range(0, 200), dd[dd['transition'] == base][metrics[i]], label = 'Pretraining')
            ax.plot(range(200, 400), dd[dd['transition'] == transitions[j]][metrics[i]], label = 'Fine-Tuning')
            ax.set_title(plot_titles[j], fontsize=10)
            ax.grid(True)
            # ax.legend()

        # Set row title for each row
        row[0].annotate(row_titles[i], xy=(0, 0.5), xytext=(-row[0].yaxis.labelpad - 5, 0),
                        xycoords=row[0].yaxis.label, textcoords='offset points',
                        size='large', ha='center', va='center', rotation=90)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, ['Pretraining', 'Fine-Tuning'], loc='upper center', bbox_to_anchor=(0.5, -0.02), ncol=3)

    plt.tight_layout()
    plt.show()

plot_sum(dd, 'node', "Trnasition from Node Level",
         ['node_to_proximity', 'node_to_cluster', 'node_to_graph'],
         ['Node to proximity', 'Node to Cluster', 'Node to graph'])

plot_sum(dd, 'proximity', "Trnasition from Proximity Level",
         ['proximity_to_node', 'proximity_to_cluster', 'proximity_to_graph'],
         ['Proximity to node', 'Proximity to cluster', 'Proximity to graph'])

plot_sum(dd, 'graph', "Trnasition from Graph Level",
         ['graph_to_node', 'graph_to_proximity', 'graph_to_cluster'],
         ['Graph to node', 'Graph to proximity', 'Graph to cluster'])

plot_sum(dd, 'cluster', "Trnasition from Cluster Level",
         ['cluster_to_node', 'cluster_to_proximity', 'cluster_to_graph'],
         ['Cluster to node', 'Cluster to proximity', 'Cluster to graph'])

#### Plots ####

def plot_results(dd, titles, transition, metric):
    fig, axs = plt.subplots(1, 4, figsize=(15, 4))
    abs = list(range(1, 401, 1))
    fig.suptitle(titles[0], fontsize=14)
    axs[0].plot(list(range(1, 201, 1)), dd[dd['transition'] == transition[0]][metric])
    axs[0].set_title(titles[1], fontsize=10)
    axs[0].grid(True)

    y1 = (dd[dd['transition'] == transition[0]][metric].to_list() + dd[dd['transition'] == transition[1]][metric].to_list())
    axs[1].vlines(200, min(y1), max(y1), color = 'red')
    axs[1].plot(abs, y1)
    axs[1].set_title(titles[2], fontsize=10)
    axs[1].grid(True)

    y2 = (dd[dd['transition'] == transition[0]][metric].to_list() + dd[dd['transition'] == transition[2]][metric].to_list())
    axs[2].vlines(200, min(y2), max(y2), color = 'red')
    axs[2].plot(abs, y2)
    axs[2].set_title(titles[3], fontsize=10)
    axs[2].grid(True)

    y3 = (dd[dd['transition'] == transition[0]][metric].to_list() + dd[dd['transition'] == transition[3]][metric].to_list())
    axs[3].vlines(200, min(y3), max(y3), color = 'red')
    axs[3].plot(abs, y3)
    axs[3].set_title(titles[4], fontsize=10)
    axs[3].grid(True)
    plt.show()


plot_results(dd, 
             ['Accuracy', 'Node level', 'Node to proximity', 'Node to Cluster', 'Node to graph'],
             ['node', 'node_to_proximity', 'node_to_cluster', 'node_to_graph'],
             'accuracy')

plot_results(dd, 
             ['Intrinsic dimension ', 'Node level', 'Node to proximity', 'Node to Cluster', 'Node to graph'],
             ['node', 'node_to_proximity', 'node_to_cluster', 'node_to_graph'],
             'ID_global_mean')

plot_results(dd, 
             ['Linear intrinsic dimension  ', 'Node level', 'Node to proximity', 'Node to Cluster', 'Node to graph'],
             ['node', 'node_to_proximity', 'node_to_cluster', 'node_to_graph'],
             'LID_mean')





plot_results(dd, 
             ['Accuracy', 'Proximity level', 'Proximity to Node', 'Proximity to Cluster', 'Proximity to Graph'],
             ['proximity', 'proximity_to_node', 'proximity_to_cluster', 'proximity_to_graph'],
             'accuracy')

plot_results(dd, 
             ['Intrinsic dimension', 'Proximity level', 'Proximity to Node', 'Proximity to Cluster', 'Proximity to Graph'],
             ['proximity', 'proximity_to_node', 'proximity_to_cluster', 'proximity_to_graph'],
             'ID_global_mean')

plot_results(dd, 
             ['Linear intrinsic dimension', 'Proximity level', 'Proximity to Node', 'Proximity to Cluster', 'Proximity to Graph'],
             ['proximity', 'proximity_to_node', 'proximity_to_cluster', 'proximity_to_graph'],
             'LID_mean')



plot_results(dd, 
             ['Accuracy', 'Cluster level', 'Cluster to Node', 'Cluster to Proximity', 'Cluster to Graph'],
             ['cluster', 'cluster_to_node', 'cluster_to_proximity', 'cluster_to_graph'],
             'accuracy')

plot_results(dd, 
             ['Intrinsic dimension', 'Cluster level', 'Cluster to Node', 'Cluster to Proximity', 'Cluster to Graph'],
             ['cluster', 'cluster_to_node', 'cluster_to_proximity', 'cluster_to_graph'],
             'ID_global_mean')

plot_results(dd, 
             ['Linear intrinsic dimension', 'Cluster level', 'Cluster to Node', 'Cluster to Proximity', 'Cluster to Graph'],
             ['cluster', 'cluster_to_node', 'cluster_to_proximity', 'cluster_to_graph'],
             'LID_mean')




plot_results(dd, 
             ['Accuracy', 'Graph level', 'Graph to Node', 'Graph to Proximity', 'Graph to Cluster'],
             ['graph', 'graph_to_node', 'graph_to_proximity', 'graph_to_cluster'],
             'accuracy')

plot_results(dd, 
             ['Intrinsic dimension', 'Graph level', 'Graph to Node', 'Graph to Proximity', 'Graph to Cluster'],
             ['graph', 'graph_to_node', 'graph_to_proximity', 'graph_to_cluster'],
             'ID_global_mean')

plot_results(dd, 
             ['Linear intrinsic dimension', 'Graph level', 'Graph to Node', 'Graph to Proximity', 'Graph to Cluster'],
             ['graph', 'graph_to_node', 'graph_to_proximity', 'graph_to_cluster'],
             'LID_mean')