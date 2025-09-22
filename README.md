# Efficient Data Subsampling for GNN Downstream Tasks

This code is the implementation of subsampling for graph data in "Efficient Data Subsampling for GNN Downstream Tasks" submitted to the 17<sup>th</sup> Asian Conference on Machine Learning (ACML) 2025. The proposed method has been evaluated on a graph classification task, but it can be extended to other tasks as well.

The dataset is a private dataset of child and youth mental health patients. A sample record for each dataset file has been included in the `sample_data` directory. You need to have two files inside `data/fused_dataset` directory called `fixed_ed.csv` and `fixed_questionnaire.csv`.

<strong>Paper:</strong> Hirad Daneshvar and Reza Samavi. "Efficient Subsampling for GNN Downstream Tasks." The 17<sup>th</sup> Asian Conference on Machine Learning (ACML), 2025.

## Setup
The code has been dockerized (using GPU). The requirements are included in the requirements.txt file. If you choose to use docker, you don't need to install the packages as it will automatically install them all. To use the docker, make sure you create a copy of _.env.example_ file and name it _.env_ and complete it according to your system. To use the dockerized version, you will need a Ubuntu based system.

If you choose to run the code using CPU, you don't need to use docker as the requirements for CPU support is included in a file called _requirements_cpu.txt_.

## Running Experiments

### The Mental Health Dataset
After creating the _.env_ file, you first need to build the image using ```docker compose build```. Then you need to run ```docker compose up -d``` to start the project. To run the experiments, you need to run the following:
- Create a `data` directory if it does not exist. Inside `data` create a directory called `fused_dataset`. Place`fused_dataset` the `csv` files in this directory.
- Preparing the data:
  ```
    docker compose exec torch bash -c "python3.9 mh_data_preparation.py"
  ```
- Training a multi-classifier GNN using the integrated dataset (self-distillation using the primary (ED) dataset):
  ```
    docker compose exec torch bash -c "python3.9 mh_run_self_dist.py"
  ```
- Uncertainty quantification (using the primary (ED) dataset):
  ```
    docker compose exec torch bash -c "python3.9 mh_compute_uncertainty.py"
  ```
- Performing clustering (using the primary (ED) dataset):
  ```
    docker compose exec torch bash -c "python3.9 mh_clustering.py"
  ```
- Performing important-based subsampling (using the primary (ED) dataset which will be used in integrating with the secondary (questionnaire) dataset):
  ```
    docker compose exec torch bash -c "python3.9 mh_sampling.py"
  ```
- Training a GNN using all training data in the integrated dataset:
  ```
    docker compose exec torch bash -c "python3.9 mh_run_all_data_fused_gnn.py"
  ```
- Training a GNN using randomly sampled data in the integrated dataset:
  ```
    docker compose exec torch bash -c "python3.9 mh_run_fused_gnn.py"
  ```
- Training a GNN using data sampled by the proposed approach (using the integrated dataset):
  ```
    docker compose exec torch bash -c "python3.9 mh_run_fused_gnn.py"
  ```
  
### The Enzymes Dataset
The dataset will be automatically downloaded and placed in the `data` directory

- Training a multi-classifier GNN using all the data in the dataset:
  ```
    docker compose exec torch bash -c "python3.9 enzymes_run_self_dist_gnn.py"
  ```
- Uncertainty quantification:
  ```
    docker compose exec torch bash -c "python3.9 enzymes_compute_uncertainty.py"
  ```
- Performing clustering:
  ```
    docker compose exec torch bash -c "python3.9 enzymes_clustering.py"
  ```
- Performing important-based subsampling:
  ```
    docker compose exec torch bash -c "python3.9 enzymes_sampling.py"
  ```
- Training a GNN using all training data:
  ```
    docker compose exec torch bash -c "python3.9 enzymes_run_all_samples.py"
  ```
- Training a GNN using data sampled by the proposed approach:
  ```
    docker compose exec torch bash -c "python3.9 enzymes_run_sampled.py"
  ```

## Generating Results
After running the experiments, the model outputs and metadata will be saved in:
- A folder called `results` in the root directory of the project

### The Mental Health Dataset
You need to run the following commands to see the results:
- Comparison of the randomly sampled data with the proposed approach (k=10):
  ```
     docker compose exec torch bash -c "python3.9 mh_results.py"
  ```
- Comparison of training using the full training dataset with the proposed approach (k=10):
  ```
     docker compose exec torch bash -c "python3.9 mh_results_all_samples.py"
  ```
- Comparison of training using the proposed approach with different number of clusters:
  ```
     docker compose exec torch bash -c "python3.9 mh_results_diff_clusters.py"
  ```
  
### The Enzymes Dataset
You need to run the following commands to see the results:
- Comparison of training using all the data with the proposed approach (k=10):
  ```
     docker compose exec torch bash -c "python3.9 enzymes_results.py"
  ```

## Cite
TBD
