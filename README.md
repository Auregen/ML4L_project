# LLMs & Commonsense Reasoning: Evaluation on WinoGrande

This repository contains the source code for our project evaluating Machine Learning models and Large Language Models (LLMs) on commonsense reasoning tasks. The primary objective is to evaluate and compare different architectural approaches to pronoun resolution using the **WinoGrande** benchmark.

## Project Overview

Pronoun resolution requires logical deduction and contextual understanding beyond simple statistical pattern recognition. Given an ambiguous sentence like *"Ann asked Mary what time the library closes, because she had forgotten"*, the model must correctly deduce whether *"she"* refers to Ann or Mary.

We implemented and compared three distinct modeling paradigms:
1. **Multi-Layer Perceptron (MLP) - Baseline:** A binary classifier trained from scratch using concatenated sentence embeddings (with a masked blank) and static candidate option embeddings.
2. **RoBERTa (Binary Fine-Tuning):** A sequence classification approach scoring independent sequences (`[sentence] [option1]` vs `[sentence] [option2]`).
3. **RoBERTa-Large (Multiple-Choice Formulation) - Final Model:** A competitive formulation where the model processes both options concurrently using the structure `[CLS] prefix c_i suffix [SEP]` to optimize cross-entropy loss over the correct choice.

## Main Results

The *Multiple-Choice* formulation using `roberta-large` significantly outperforms independent sequence classification methods and achieves state-of-the-art parity on the validation split ($N = 1267$).

| Model | Accuracy |
| :--- | :--- |
| Multi-Layer Perceptron (Ours) | 50.99% |
| RoBERTa Zero-shot | 52.49% |
| RoBERTa Binary Fine-tuned (Ours) | 50.59% |
| **RoBERTa Multiple-Choice (Ours)** | **78.69%** |
| RoBERTa Fine-tuned (Sakaguchi Reference) | 79.72% |

### Statistical Significance

A paired exact McNemar's test demonstrates that the Multiple-Choice model significantly outperforms the baseline MLP ($p = 1.34 \times 10^{-43}$). Furthermore, when compared against the reference Sakaguchi model, the test indicates 82 instances where our model was uniquely correct and 95 instances where the reference model was uniquely correct. The difference is not statistically significant ($p = 0.367$), confirming that our model achieves performance parity with the established reference on this split.

### Technical Observation & Limitations

While our final model reaches a high evaluation accuracy, training metrics indicate an open discrepancy between the training loss (0.215) and evaluation loss (1.505). This pronounced gap highlights a high level of overfitting, suggesting that the model relies partly on memorized training cues and residual dataset biases rather than clean logical deduction.

## Repository Structure

├── configs/             # YAML configuration files for hyperparameter tracking
├── data/
│   ├── raw/             # Raw WinoGrande dataset splits (train/validation/test)
│   └── processed/       # Placeholders for processed or tokenized artifacts
├── notebooks/           # Exploratory notebooks and baseline MLP training script
├── outputs/             # (Auto-generated) Checkpoints, predictions, and text reports
│   ├── models/
│   ├── predictions/
│   └── reports/
├── src/                 # Main source code
│   ├── data_prep.py     # Data parsing, splitting, and Hugging Face dataset preparation
│   ├── train_mcq.py     # Main multiple-choice fine-tuning pipeline via the Trainer API
│   ├── evaluate_model.py# Inference script generating CSV predictions and metrics
│   └── io_utils.py      # Standard file I/O utilities (YAML, JSON)
├── stats/               # Statistical significance tools
│   └── mcnemar.py       # Computation of contingent tables, chi2, and exact p-values
├── tests/               # Unit testing suite (pytest)
└── tools/               # Operational shell scripts


## Installation

Clone this repository and install the frozen package dependencies. Utilizing a virtual environment is highly recommended.

git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
cd your-repo-name
pip install -r requirements.txt


*Note on Apple Silicon compatibility:* The scripts explicitly handle the `mps` device fallback via internal environment definitions, preventing execution errors caused by local background framework overrides.

## Usage and Reproducibility

### 1. End-to-End Execution (Training & Evaluation)

To replicate the final Multiple-Choice model using our optimized hyperparameters, run:

bash tools/run_experiment.sh


This wrapper script reads parameters from `configs/experiment.yaml`, launches model fine-tuning, executes verification against the validation split, and dumps performance files inside `outputs/reports/`.

### 2. Manual Evaluation

To isolate inference on a pre-trained checkpoint and output explicit predictions along with an evaluation summary:

python -m src.evaluate_model \
  --model-dir outputs/models/run_mcq \
  --data-dir data/raw \
  --split validation \
  --pred-dir outputs/predictions \
  --report-dir outputs/reports


### 3. Statistical Testing (McNemar)

To evaluate the statistical divergence between two models using paired outputs:

python stats/mcnemar.py \
  --pred-a outputs/predictions/pred_validation_mlp.csv \
  --pred-b outputs/predictions/pred_validation_roberta_mcq.csv \
  --out outputs/reports/mcnemar_report.txt


## Unit Tests

To run verification assertions regarding text segmentation, label mapping, and statistical computations:

pytest -q


## Contributors

* Nélia Bouzid
* Thomas Ben Yazza
* Aurélien Valdecasa

*Project developed as part of the ML4L course requirements.*
