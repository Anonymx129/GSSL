# Smooth Transitions in Graph Self-Supervision: Mitigating Feature Twist Across Abstraction Levels

This repository contains the implementation of my master’s degree research project. We propose a self-supervised framework for learning graph representations at multiple levels: node, proximity, cluster, and graph levels. The framework aims to push the boundaries of graph representation learning by achieving state-of-the-art results and exploring the transitions between abstraction levels in graph data.

## Project Overview

### Key Objectives

1. **State-of-the-Art Performance:**
   - Achieve experimental results at each level (node, proximity, cluster, and graph) that are comparable to or better than existing state-of-the-art methods.

2. **Understanding Abstraction Transitions:**
   - Explore the transition between abstraction levels using:
     - Intrinsic Dimension (ID)
     - Local Intrinsic Dimensionality (LID)
     - Classification and clustering accuracy as downstream tasks

3. **Mitigating Feature Twist Effect:**
   - Address the "feature twist effect" that arises during transitions between abstraction levels.
   - Improve latent space representation to enhance performance in downstream tasks.

### Framework Highlights
- Self-supervised learning approach to train graph representations.
- Evaluation at multiple levels of abstraction.
- Tools for analyzing transitions between abstraction levels.
- Focus on improving downstream task performance.

## Installation
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## How to Use


### Hyperparameter Tuning
Experiment with different hyperparameters to optimize results.

### Evaluations
You can evaluate the framework using pre-defined evaluation functions. These functions are also available as Jupyter Notebook files for interactive analysis:
```bash
python evaluate.py --level <node|proximity|cluster|graph>
```
Or open the corresponding notebook file for interactive exploration:
```bash
jupyter notebook evaluations.ipynb
```

## Directory Structure
```
.
├── parameters/                   # Best parameters saved
├── pretrained_models/            # Saved models and weights
├── models/                       # Source code for the framework
├── saved_results/                # Directory for storing evaluation results
├── first_eval_protocol_cora.py   # Evaluate model performance and transition between abstration levels
├── eval_proximity_imp.py         # Evaluate smooth transition from node to proximity 
├── eval_cluster_imp.py           # Evaluate smooth transition from node to cluster 
├── data_stats.py                 # Generate data stats
├── requirements.txt              # Python dependencies
└── README.md                     # Project README file
```

## Experimental Results
We provide experimental results for each level of abstraction. Key metrics include:
- Intrinsic Dimension (ID)
- Local Intrinsic Dimensionality (LID)
- Classification accuracy
- Clustering accuracy

## Acknowledgments
This research is part of my master’s degree project. I extend my gratitude to my academic advisors and peers for their invaluable guidance and support.