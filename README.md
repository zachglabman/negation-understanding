# Attention is *Not* All You Need: Negation Understanding in Transformers

## Description:
This repository contains code and resources for the paper "Attention is *Not* All You Need: Negation Understanding in Transformers". The paper investigates the understanding of negation in transformer-based language models, specifically focusing on the impact of data augmentation on model performance in recognizing textual entailment (RTE) tasks. This research was motivated by the question of whether or not "negation-aware" language models like those in Helwe et al. (2022) and Hosseini et al. (2021) actually understand negation.

## Instructions:
- Clone the repository
- Install dependencies by typing `pip install -r requirements.txt` into the terminal

## Repository Structure
- `outputs/`: Contains both Gram and Full model predictions in JSON format as well as text-file outputs of models evaluated on the MNLI test set.
- `train_and_eval/`: Contains the Python files necessary to create grammatical and negated data, as well as tune hyperparameters, train and save models.
- `background_and_models.py`: Contains the code necessary to run files within `train_and_eval/` as well as to visualize the models.
- `visualize_models.ipynb`: A Jupyter notebook visualizing the train and test data distributions, the models, and their predictions in various ways, as described in the paper.
- `Paper.pdf`: The research paper associated with this repository.

## Data
The MNLI dataset (Williams et al. 2018) used in this study can be downloaded from [here](https://cims.nyu.edu/~sbowman/multinli/).

## Usage
1. Download the MNLI dataset and place it in the appropriate directory.
2. Run the scripts in the `train_and_eval/` directory to create the datasets, train the models, and evaluate them on the MNLI test set.
3. Visualize models using the jupyter notebook provided.
