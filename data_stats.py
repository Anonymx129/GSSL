import argparse
import warnings

from torch_geometric.datasets import Planetoid, CitationFull
from torch_geometric.utils import to_networkx

warnings.filterwarnings('ignore')


def load_dataset(name, root):
    """
    Load a dataset by name.
    """
    if name.lower() == 'dblp':
        return CitationFull(root=root, name='dblp')
    elif name.lower() in ['pubmed', 'cora', 'citeseer']:
        return Planetoid(root=root, name=name.capitalize())
    else:
        raise ValueError(f"Unsupported dataset name: {name}")


def data_stats(data, dataset_name, dataset):
    """
    Print statistics for the dataset.
    """
    print(f"Dataset: {dataset_name}")
    print(f"Number of nodes: {data.num_nodes}")
    print(f"Number of edges: {data.num_edges}")
    print(f"Has isolated nodes: {data.has_isolated_nodes()}")
    print(f"Has self-loops: {data.has_self_loops()}")
    print(f"Is undirected: {data.is_undirected()}")
    print(f"Average node degree: {data.num_edges / data.num_nodes:.2f}")
    print(f"Number of features: {data.num_node_features}")
    print(f"Number of edge features: {data.num_edge_features}")
    print(f"Number of classes: {dataset.num_classes}")
    print("-" * 40)


def main(args):
    for dataset_name in args.datasets:
        try:
            dataset = load_dataset(dataset_name, args.root)
            data = dataset[0]
            data_stats(data, dataset_name, dataset)
        except Exception as e:
            print(f"Error loading dataset {dataset_name}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset Statistics Viewer")
    parser.add_argument(
        '--datasets',
        type=str,
        nargs='+',
        default=['dblp', 'pubmed', 'cora', 'citeseer'],
        help="List of dataset names to load (e.g., 'dblp', 'pubmed', 'cora', 'citeseer').",
    )
    parser.add_argument(
        '--root',
        type=str,
        default='/tmp',
        help="Root directory for storing datasets.",
    )
    args = parser.parse_args()
    main(args)