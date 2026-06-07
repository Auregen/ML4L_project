# =============================================================================
# Imports and Environment Setup
# =============================================================================
import os
import time
import datetime
import warnings

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from torch.utils.data import DataLoader
import evaluate
from scipy.stats import chi2_contingency

# Suppress warnings for cleaner console output
warnings.filterwarnings('ignore')

# Detect available hardware accelerator
# Supports Apple Silicon (MPS), NVIDIA GPUs (CUDA), and fallback to CPU
if torch.backends.mps.is_available():
    device = torch.device('mps')
elif torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print(f'Device : {device}')


# =============================================================================
# Configuration and Hyperparameters
# =============================================================================
TRAIN_CONFIG    = 'winogrande_xl'
BASE_MODEL      = 'roberta-base'
OUR_MODEL_PATH  = './notre_modele'
SAKAGUCHI_MODEL = 'DeepPavlov/roberta-large-winogrande' # Reference model

MAX_LENGTH      = 128
EPOCHS          = 3
LEARNING_RATE   = 2e-5
BATCH_SIZE      = 16
WEIGHT_DECAY    = 0.01

# Create necessary directories for model checkpoints and prediction outputs
os.makedirs(OUR_MODEL_PATH, exist_ok=True)
os.makedirs('./predictions', exist_ok=True)
print('Config OK.')


# =============================================================================
# Data Loading and Baseline Computation
# =============================================================================
# Load the WinoGrande dataset from HuggingFace
dataset   = load_dataset('winogrande', TRAIN_CONFIG)
train_raw = dataset['train']
dev_raw   = dataset['validation']

print(f'Train : {len(train_raw)} examples')
print(f'Dev   : {len(dev_raw)} examples')

# Display a sample to understand the dataset structure
ex = train_raw[0]
print(f"\nExample — sentence : {ex['sentence']}")
print(f"           option1  : {ex['option1']}")
print(f"           option2  : {ex['option2']}")
print(f"           answer   : {ex['answer']} ({'option1' if ex['answer']=='1' else 'option2'} is correct)")

# Calculate the majority class baseline for the development set
dev_answers = dev_raw['answer']
n1 = dev_answers.count('1')
n2 = dev_answers.count('2')
print(f'\nDev — option1 correct : {n1} ({n1/len(dev_answers)*100:.1f}%)')
print(f'Dev — option2 correct : {n2} ({n2/len(dev_answers)*100:.1f}%)')
print(f'Majority baseline     : {max(n1,n2)/len(dev_answers)*100:.1f}%')


# =============================================================================
# Data Preprocessing
# WinoGrande uses a fill-in-the-blank format with an underscore '_'.
# We transform this multiple-choice task into a binary classification task.
# =============================================================================

def split_sentence(sentence):
    """
    Splits the sentence around the blank marker '_'.
    Returns the substring before the blank and the substring after it.
    """
    parts  = sentence.split('_')
    before = parts[0].rstrip()
    after  = parts[1].lstrip() if len(parts) > 1 else ''
    return before, after


def hf_dataset_to_binary_pairs(hf_dataset, tokenizer, max_length=MAX_LENGTH):
    """
    Converts a WinoGrande dataset into binary classification pairs.
    Each single WinoGrande example yields two separate binary examples
    (one for option1, one for option2). 
    The label is 1 if the option is correct, and 0 otherwise.
    """
    input_ids_list      = []
    attention_mask_list = []
    labels_list         = []

    for ex in tqdm(hf_dataset, desc='Tokenization', leave=False):
        before, after = split_sentence(ex['sentence'])
        
        # Process both options independently
        for option, option_id in [(ex['option1'], '1'), (ex['option2'], '2')]:
            # Reconstruct the sentence by filling the blank with the current option
            text_b = f"{option} {after}".strip()
            
            # Tokenize as a sentence pair: [CLS] Context before blank [SEP] Option + Context after blank [SEP]
            enc = tokenizer(
                before, text_b,
                truncation=True,
                max_length=max_length,
                padding='max_length',
            )
            input_ids_list.append(enc['input_ids'])
            attention_mask_list.append(enc['attention_mask'])
            
            # Assign label 1 if this option matches the ground truth answer, else 0
            labels_list.append(1 if ex['answer'] == option_id else 0)

    return Dataset.from_dict({
        'input_ids':      input_ids_list,
        'attention_mask': attention_mask_list,
        'label':          labels_list,
    })


tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

print('\nTokenizing the training split...')
train_dataset = hf_dataset_to_binary_pairs(train_raw, tokenizer)
print(f'  → {len(train_dataset)} pairs ({len(train_raw)} × 2)')

print('Tokenizing the dev split...')
dev_dataset = hf_dataset_to_binary_pairs(dev_raw, tokenizer)
print(f'  → {len(dev_dataset)} pairs ({len(dev_raw)} × 2)')

train_labels = train_dataset['label']
print(f'\nTrain — label 1 (correct)  : {train_labels.count(1)} ({train_labels.count(1)/len(train_labels)*100:.1f}%)')
print(f'Train — label 0 (incorrect) : {train_labels.count(0)} ({train_labels.count(0)/len(train_labels)*100:.1f}%)')


# =============================================================================
# Training Time Estimation (Dry Run)
# Executes a few forward/backward passes to estimate total training duration.
# =============================================================================

def _collate(batch):
    """Custom collate function to stack tensors for the DataLoader."""
    return {
        'input_ids':      torch.tensor([b['input_ids']      for b in batch]),
        'attention_mask': torch.tensor([b['attention_mask'] for b in batch]),
        'labels':         torch.tensor([b['label']          for b in batch]),
    }

N_BENCH = 6   # Number of steps to measure (first step ignored to account for GPU/MPS warmup)

# Initialize a dummy model and optimizer strictly for benchmarking
_m   = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=2)
_m.to(device).train()
_opt = torch.optim.AdamW(_m.parameters(), lr=LEARNING_RATE)
_ldr = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=_collate)

_times = []
for _i, _b in enumerate(_ldr):
    if _i >= N_BENCH:
        break
    _b = {k: v.to(device) for k, v in _b.items()}
    _t = time.perf_counter()
    _m(**_b).loss.backward()
    _opt.step()
    _opt.zero_grad()
    _times.append(time.perf_counter() - _t)

del _m, _opt, _ldr # Free up memory before actual training

# Calculate average time per step, ignoring the first warmup iteration
_avg          = sum(_times[1:]) / len(_times[1:])    
_steps_epoch  = len(train_dataset) // BATCH_SIZE
_total_steps  = _steps_epoch * EPOCHS
_total_sec    = _avg * _total_steps
_eta          = datetime.datetime.now() + datetime.timedelta(seconds=_total_sec)

hh = int(_total_sec // 3600)
mm = int((_total_sec % 3600) // 60)
ss = int(_total_sec % 60)

print('\n' + '═' * 52)
print(f'  Time / step    : {_avg:.2f}s  (batch_size = {BATCH_SIZE})')
print(f'  Steps / epoch  : {_steps_epoch}')
print(f'  Epochs         : {EPOCHS}')
print(f'  Total steps    : {_total_steps}')
print(f'  Estimated time : {hh}h {mm}min {ss}s')
print(f'  ETA            : {_eta.strftime("%H:%M:%S")}')
print('═' * 52 + '\n')


# =============================================================================
# Model Fine-tuning (Sequence Classification)
# =============================================================================

accuracy_metric = evaluate.load('accuracy')

def compute_metrics(eval_pred):
    """Calculates accuracy during the Trainer evaluation phase."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return accuracy_metric.compute(predictions=preds, references=labels)

# Initialize the actual model to be trained with a binary classification head
model_ft = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=2)

training_args = TrainingArguments(
    output_dir=OUR_MODEL_PATH,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    eval_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,
    metric_for_best_model='accuracy',
    logging_steps=50,
    report_to='none', 
    # Note: MPS (Apple Silicon) or CUDA is automatically handled by the Accelerate backend
)

trainer = Trainer(
    model=model_ft,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    compute_metrics=compute_metrics,
)

print('Starting training...')
train_result = trainer.train()

print(f'\n✓ Done!')
print(f'  Final Loss : {train_result.training_loss:.4f}')

# Save the fine-tuned model and its tokenizer
trainer.save_model(OUR_MODEL_PATH)
tokenizer.save_pretrained(OUR_MODEL_PATH)
print(f'  Model saved to : {OUR_MODEL_PATH}')

# Extract and display evaluation metrics per epoch
log_history = pd.DataFrame(trainer.state.log_history)
eval_logs   = log_history[log_history['eval_accuracy'].notna()][['epoch', 'eval_accuracy', 'eval_loss']]
print('\nResults per epoch (evaluated on binary dev pairs):')
print(eval_logs.to_string(index=False))


# =============================================================================
# Evaluation Methods Setup
# =============================================================================

def get_true_labels(hf_dataset):
    """Maps the WinoGrande string answers ('1' or '2') to binary integers (0 or 1)."""
    return [0 if ex['answer'] == '1' else 1 for ex in hf_dataset]


def evaluate_binary_classifier(model, tokenizer, hf_dataset, device, max_length=MAX_LENGTH):
    """
    Evaluates a model fine-tuned for Sequence Classification on WinoGrande.
    For each problem, it computes the probability P(correct=1) for both options independently.
    The model predicts the option that yields the higher positive class probability.
    """
    model.eval().to(device)
    predictions = []

    for ex in tqdm(hf_dataset, desc='Evaluating Classifier', leave=False):
        before, after = split_sentence(ex['sentence'])
        scores = []

        # Evaluate both options
        for option in [ex['option1'], ex['option2']]:
            text_b = f"{option} {after}".strip()
            enc = tokenizer(
                before, text_b,
                truncation=True,
                max_length=max_length,
                return_tensors='pt',
            ).to(device)
            
            with torch.no_grad():
                out = model(**enc)
            # Store the softmax probability of the 'correct' class (index 1)
            scores.append(torch.softmax(out.logits[0], dim=-1)[1].item())

        # Predict option 1 (index 0) if it has a higher score, else option 2 (index 1)
        predictions.append(0 if scores[0] > scores[1] else 1)

    true_labels = get_true_labels(hf_dataset)
    accuracy = sum(p == t for p, t in zip(predictions, true_labels)) / len(true_labels)
    return accuracy, predictions, true_labels


def score_option_mlm(model, tokenizer, sentence, option, device, max_length=MAX_LENGTH):
    """
    Computes the log-probability of a specific option replacing the blank `_`,
    using a pre-trained Masked Language Model (zero-shot, no fine-tuning).
    """
    masked  = sentence.replace('_', tokenizer.mask_token)
    inputs  = tokenizer(masked, truncation=True, max_length=max_length, return_tensors='pt').to(device)
    
    # Locate the position of the mask token in the input sequence
    mask_pos = (inputs.input_ids[0] == tokenizer.mask_token_id).nonzero(as_tuple=True)[0]
    if len(mask_pos) == 0:
        return 0.0
        
    # Tokenize the target option (adding a leading space is often required for RoBERTa tokenization)
    option_ids = tokenizer.encode(f' {option}', add_special_tokens=False)
    if not option_ids:
        return 0.0
        
    with torch.no_grad():
        out = model(**inputs)
        
    # Return the log-softmax probability of the first token of the option at the mask position
    return torch.log_softmax(out.logits[0, mask_pos[0].item()], dim=-1)[option_ids[0]].item()


def evaluate_mlm_baseline(model, tokenizer, hf_dataset, device):
    """
    Evaluates the zero-shot performance of a base model using MLM scoring.
    Predicts the option that maximizes the token probability at the masked position.
    """
    model.eval().to(device)
    predictions = []
    
    for ex in tqdm(hf_dataset, desc='Evaluating MLM Baseline', leave=False):
        s1 = score_option_mlm(model, tokenizer, ex['sentence'], ex['option1'], device)
        s2 = score_option_mlm(model, tokenizer, ex['sentence'], ex['option2'], device)
        predictions.append(0 if s1 > s2 else 1)
        
    true_labels = get_true_labels(hf_dataset)
    accuracy = sum(p == t for p, t in zip(predictions, true_labels)) / len(true_labels)
    return accuracy, predictions, true_labels


# =============================================================================
# Model Benchmarking Pipeline
# Evaluates 3 distinct models to compare their performance.
# =============================================================================

# Model 1: Zero-shot baseline using pre-trained RoBERTa (Masked Language Modeling)
print('\n── Model 1 : RoBERTa baseline (MLM zero-shot) ──')
tokenizer_base = AutoTokenizer.from_pretrained(BASE_MODEL)
model_baseline = AutoModelForMaskedLM.from_pretrained(BASE_MODEL)
acc_baseline, preds_baseline, true_labels = evaluate_mlm_baseline(
    model_baseline, tokenizer_base, dev_raw, device
)
print(f'Baseline Accuracy : {acc_baseline:.4f}  ({acc_baseline*100:.2f}%)')

# Model 2: Our newly fine-tuned RoBERTa model (Sequence Classification)
print('\n── Model 2 : Our fine-tuned RoBERTa ──')
tokenizer_ft  = AutoTokenizer.from_pretrained(OUR_MODEL_PATH)
model_ft_eval = AutoModelForSequenceClassification.from_pretrained(OUR_MODEL_PATH)
acc_ft, preds_ft, _ = evaluate_binary_classifier(
    model_ft_eval, tokenizer_ft, dev_raw, device
)
print(f'Our Model Accuracy : {acc_ft:.4f}  ({acc_ft*100:.2f}%)')

# Model 3: A community reference model (Sakaguchi's roberta-large fine-tuned on WinoGrande)
print('\n── Model 3 : roberta-large-winogrande (Reference) ──')
tokenizer_sak = AutoTokenizer.from_pretrained(SAKAGUCHI_MODEL)
model_sak     = AutoModelForSequenceClassification.from_pretrained(SAKAGUCHI_MODEL)
acc_sak, preds_sak, _ = evaluate_binary_classifier(
    model_sak, tokenizer_sak, dev_raw, device
)
print(f'Reference Accuracy : {acc_sak:.4f}  ({acc_sak*100:.2f}%)')


# =============================================================================
# Comparative Results & Statistical Testing
# =============================================================================

def delta(acc, ref):
    """Calculates the percentage point difference between two accuracies."""
    d    = acc - ref
    sign = '+' if d >= 0 else ''
    return f'{sign}{d*100:.2f}pp'

# Format the results into a clean pandas DataFrame for display
results = pd.DataFrame([
    {
        'Model':          'roberta-base (baseline MLM)',
        'Architecture':   'roberta-base',
        'Fine-tuning':    'No',
        'Accuracy':       f'{acc_baseline*100:.2f}%',
        'Δ vs baseline':  '—',
    },
    {
        'Model':          'Our fine-tuned roberta-base',
        'Architecture':   'roberta-base',
        'Fine-tuning':    f'Yes ({TRAIN_CONFIG}, {EPOCHS} epochs)',
        'Accuracy':       f'{acc_ft*100:.2f}%',
        'Δ vs baseline':  delta(acc_ft, acc_baseline),
    },
    {
        'Model':          'roberta-large-winogrande (reference)',
        'Architecture':   'roberta-large',
        'Fine-tuning':    'Yes (Full WinoGrande)',
        'Accuracy':       f'{acc_sak*100:.2f}%',
        'Δ vs baseline':  delta(acc_sak, acc_baseline),
    },
])

print('\n' + results.to_string(index=False))


def mcnemar_test(preds_a, preds_b, true_labels, name_a, name_b, alpha=0.05):
    """
    Performs McNemar's test for paired nominal data to determine if the difference
    in predictions between two models is statistically significant.
    
    Contingency table components:
      b00: Both models correct
      b01: Model A correct, Model B incorrect
      b10: Model A incorrect, Model B correct
      b11: Both models incorrect
    """
    b00 = b01 = b10 = b11 = 0
    for pa, pb, t in zip(preds_a, preds_b, true_labels):
        ca, cb = (pa == t), (pb == t)
        if   ca and     cb: b00 += 1
        elif ca and not cb: b01 += 1
        elif not ca and cb: b10 += 1
        else:               b11 += 1

    # Calculate chi-squared statistic and p-value
    chi2, p, _, _ = chi2_contingency([[b00, b01], [b10, b11]], correction=True)

    sep = '=' * 55
    print(f'\n{sep}')
    print(f'McNemar Test : {name_a}  vs  {name_b}')
    print(sep)
    print(f'  Both correct         (b00) : {b00}')
    print(f'  Only {name_a[:10]:<10} correct (b01) : {b01}')
    print(f'  Only {name_b[:10]:<10} correct (b10) : {b10}')
    print(f'  Both incorrect       (b11) : {b11}')
    print(f'  chi2 = {chi2:.4f}   p-value = {p:.4f}')
    
    if p < alpha:
        print(f'  ✓ STATISTICALLY SIGNIFICANT difference (p < {alpha})')
    else:
        print(f'  ✗ Difference NOT statistically significant (p >= {alpha})')
    return p

# Run statistical tests comparing the three models against each other
p1 = mcnemar_test(preds_baseline, preds_ft,  true_labels, 'Baseline',  'Our FT')
p2 = mcnemar_test(preds_baseline, preds_sak, true_labels, 'Baseline',  'Reference')
p3 = mcnemar_test(preds_ft,       preds_sak, true_labels, 'Our FT',  'Reference')


# =============================================================================
# Error Analysis & Data Export
# =============================================================================

def show_disagreements(preds_a, preds_b, true_labels, hf_dataset, name_a, name_b, n=5):
    """
    Extracts and prints specific examples where the two models disagree.
    Useful for qualitative error analysis.
    """
    a_wins, b_wins = [], []
    for i, (pa, pb, t) in enumerate(zip(preds_a, preds_b, true_labels)):
        if pa == t and pb != t:
            a_wins.append(i)
        elif pa != t and pb == t:
            b_wins.append(i)

    # Print a few samples from each disagreement category
    for label, indices in [(f'{name_a} correct, {name_b} wrong', a_wins),
                           (f'{name_a} wrong, {name_b} correct', b_wins)]:
        print(f'\n── {label} ({len(indices)} examples) ──')
        for i in indices[:n]:
            ex      = hf_dataset[i]
            correct = ex['option1'] if ex['answer'] == '1' else ex['option2']
            wrong   = ex['option2'] if ex['answer'] == '1' else ex['option1']
            print(f'  [{i:4d}] {ex["sentence"]}')
            print(f'         correct={correct!r}  wrong={wrong!r}')


show_disagreements(preds_ft, preds_sak, true_labels, dev_raw, 'Our FT', 'Reference')


def save_preds(preds, true_labels, hf_dataset, path):
    """Saves the inference results and raw sentences to a CSV file."""
    rows = []
    for i, (p, t) in enumerate(zip(preds, true_labels)):
        ex = hf_dataset[i]
        rows.append({
            'idx':              i,
            'sentence':         ex['sentence'],
            'option1':          ex['option1'],
            'option2':          ex['option2'],
            'answer':           ex['answer'],
            'correct_option':   ex['option1'] if ex['answer'] == '1' else ex['option2'],
            'predicted_option': ex['option1'] if p == 0 else ex['option2'],
            'correct':          int(p == t),
        })
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f'Saved : {path}')

# Export predictions for further offline analysis
save_preds(preds_baseline, true_labels, dev_raw, './predictions/baseline.csv')
save_preds(preds_ft,       true_labels, dev_raw, './predictions/notre_ft.csv')
save_preds(preds_sak,      true_labels, dev_raw, './predictions/reference_sak.csv')


# =============================================================================
# Final Execution Summary
# =============================================================================
print('\n' + '═' * 52)
print('FINAL SUMMARY')
print('═' * 52)
print(f'Baseline RoBERTa          : {acc_baseline*100:.2f}%')
print(f'Our fine-tuned RoBERTa    : {acc_ft*100:.2f}%  (Δ = {delta(acc_ft, acc_baseline)})')
print(f'Reference (roberta-large) : {acc_sak*100:.2f}%  (Δ = {delta(acc_sak, acc_baseline)})')
print()
print('Hypothesis: Fine-tuning significantly improves performance.')
print(f'→ Baseline vs Our FT : p = {p1:.4f}  →  {"CONFIRMED" if p1 < 0.05 else "REJECTED"}')
print('═' * 52)