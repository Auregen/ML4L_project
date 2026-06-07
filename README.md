# LLMs & Commonsense Reasoning: Evaluation on WinoGrande

This repository contains the source code for our project evaluating Machine Learning models and Large Language Models (LLMs) on commonsense reasoning tasks. The primary objective is to evaluate and compare different architectural approaches to pronoun resolution using the WinoGrande benchmark.

## Project Overview

Pronoun resolution requires logical deduction and contextual understanding beyond simple statistical pattern recognition. Given an ambiguous sentence like "Ann asked Mary what time the library closes, because she had forgotten", the model must correctly deduce whether "she" refers to Ann or Mary.

We implemented and compared four distinct modeling paradigms:
1. Multi-Layer Perceptron (MLP) Baseline: A binary classifier trained from scratch using concatenated sentence embeddings (with a masked blank) and static candidate option embeddings.
2. RoBERTa-Base Zero-Shot: A baseline evaluation using standard Masked Language Modeling (MLM) scoring without any task-specific fine-tuning.
3. RoBERTa-Base (Binary Fine-Tuning): A sequence classification approach scoring independent sequences ("[sentence] [option1]" vs "[sentence] [option2]").
4. RoBERTa-Large (Multiple-Choice Formulation) Final Model: A competitive formulation where the model processes both options concurrently using the structure "[CLS] prefix c_i suffix [SEP]" to optimize cross-entropy loss over the correct choice.

## Main Results

The Multiple-Choice formulation using roberta-large significantly outperforms independent sequence classification methods and achieves state-of-the-art parity on the validation split (N = 1267).

| Model | Accuracy |
| :--- | :--- |
| Multi-Layer Perceptron (MLP Baseline) | 50.99% |
| roberta-base Zero-shot (MLM) | 52.49% |
| roberta-base Binary Fine-tuned | 50.59% |
| roberta-large Multiple-Choice (Ours) | 78.69% |
| roberta-large Fine-tuned (Sakaguchi Reference) | 79.72% |


### Technical Observation & Limitations

While our final model reaches a high evaluation accuracy, training metrics indicate a discrepancy between the training loss (0.215) and evaluation loss (1.505). This pronounced gap highlights a degree of overfitting, suggesting that the model relies partly on memorized training cues and residual dataset biases rather than purely abstract logical deduction.

## Repository Structure

```text
├── configs/               # YAML configuration files for hyperparameter tracking
├── data/
│   ├── raw/               # Raw WinoGrande dataset splits (train/validation/test)
│   └── processed/         # Placeholders for processed or tokenized artifacts
├── mlp/                   # Multi-Layer Perceptron baseline implementation
│   └── MLP_1.ipynb        # Notebook containing MLP architecture definition and training
├── binary_baseline/       # Legacy code for the roberta-base binary classification
│   └── train_binary.py    # Training and MLM scoring script for sequence classification
├── outputs/               # (Auto-generated) Checkpoints, predictions, and text reports
│   ├── models/
│   ├── predictions/
│   └── reports/
├── src/                   # Main multiple-choice pipeline source code
│   ├── data_prep.py       # Data parsing, splitting, and Hugging Face dataset preparation
│   ├── train_mcq.py       # Main multiple-choice fine-tuning pipeline via the Trainer API
│   ├── evaluate_model.py  # Inference script generating CSV predictions and metrics
│   └── io_utils.py        # Standard file I/O utilities (YAML, JSON)
├── stats/                 # Statistical significance tools
│   └── mcnemar.py         # Computation of contingent tables, chi2, and exact p-values
├── tests/                 # Unit testing suite (pytest)
└── tools/                 # Operational shell scripts
    └── run_experiment.sh  # End-to-end execution wrapper

```

## Installation

Clone this repository and install the frozen package dependencies. Utilizing a Python virtual environment is highly recommended.

```bash
git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
cd your-repo-name
pip install -r requirements.txt

```

Note on Apple Silicon compatibility: The scripts explicitly handle the mps device fallback via internal environment definitions, preventing execution errors caused by local background framework overrides.

## Usage and Reproducibility

### 1. End-to-End Execution (Training & Evaluation)

To replicate the final Multiple-Choice model using our optimized hyperparameters, run:

```bash
bash tools/run_experiment.sh

```

This wrapper script reads parameters from configs/experiment.yaml, launches model fine-tuning, executes verification against the validation split, and dumps performance metrics inside outputs/reports/.

### 2. Manual Evaluation

To isolate inference on a pre-trained checkpoint and output explicit predictions along with an evaluation summary:

```bash
python -m src.evaluate_model \
  --model-dir outputs/models/run_mcq \
  --data-dir data/raw \
  --split validation \
  --pred-dir outputs/predictions \
  --report-dir outputs/reports

```

### 3. Statistical Testing (McNemar)

To evaluate the statistical divergence between two models using paired prediction outputs:

```bash
python stats/mcnemar.py \
  --pred-a outputs/predictions/pred_validation_mlp.csv \
  --pred-b outputs/predictions/pred_validation_roberta_mcq.csv \
  --out outputs/reports/mcnemar_report.txt

```

## Unit Tests

To run verification assertions regarding text segmentation, label mapping, and statistical computations:

```bash
pytest -q

```

## Contributors

* Nélia Bouzid
* Thomas Ben Yazza
* Aurélien Valdecasa

Project developed as part of the ML4L course requirements.
