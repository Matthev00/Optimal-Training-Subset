# Optimal Training Subset

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

The goal of the project is to identify a subset of the most representative examples from each class in an image classification problem (e.g., datasets like FashionMNIST, CIFAR-10, or CIFAR-100). The objective is to determine which images are sufficient to train a well-performing classifier that ensures optimal separation between classes.

## Instalation
1. Clone repository  
    ` git clone https://gitlab-stud.elka.pw.edu.pl/mostasze/optimal_training_subset.git`
2. Prepare enviroment   
    `make_venv`
3. Install requirements   
    `make requirements`

## Running experiments
In order too replicate experiments run     
    `make run_experiments`.  
To inspect results run   
    `mlflow ui`.


```
├── Makefile           <- Makefile with convenience commands
├── README.md          <- The top-level README for developers using this project.
├── data
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- Documents
│
├── notebooks          <- Jupyter notebooks. Initial experiments.
│
├── pyproject.toml   
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports          
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment
│
├── setup.cfg          <- Configuration file for flake8
│
└── optimal_training_subset   <- Source code for use in this project.
    │
    ├── data           <- Data management and loading
    │
    ├── evolutionary   <- Evolutionary strategies implementations
    │
    ├── experiments    <- Experiment scripts
    │
    ├── models         <- Model definitions and architectures
    │
    ├── optimizers     <- Hill climbing algorithms
    │
    ├── utils          <- Utility functions and helpers
    │   
    └── config.py      <- Configuration settings

```

--------

## Authors
Mateusz Ostaszewski  
Michał Sadowski