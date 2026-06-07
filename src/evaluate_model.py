"""
Model evaluation script for a multiple-choice task (e.g., WinoGrande).
This script loads a fine-tuned model, processes the evaluation dataset,
generates predictions, and computes accuracy if gold labels are available.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForMultipleChoice, AutoTokenizer

from src.data_prep import answer_to_label, load_local_winogrande, split_sentence
from src.io_utils import ensure_dir, save_json


def parse_args() -> argparse.Namespace:
    """
    Parses command-line arguments for the evaluation script.

    Returns:
        argparse.Namespace: The parsed arguments containing directory paths,
        split choices, and tokenization constraints.
    """
    parser = argparse.ArgumentParser(description="Multiple-choice model evaluation")
    parser.add_argument("--model-dir", type=str, required=True, help="Directory containing the fine-tuned model")
    parser.add_argument("--data-dir", type=str, default="data/raw", help="Directory containing the raw CSV data files")
    parser.add_argument("--split", type=str, default="validation", choices=["validation", "test"], help="Dataset split to evaluate on")
    parser.add_argument("--max-length", type=int, default=160, help="Maximum sequence length for tokenization")
    parser.add_argument("--pred-dir", type=str, default="outputs/predictions", help="Directory to save prediction CSV files")
    parser.add_argument("--report-dir", type=str, default="outputs/reports", help="Directory to save evaluation reports (JSON)")
    return parser.parse_args()


def predict_one(model, tokenizer, sentence: str, option1: str, option2: str, max_length: int, device: torch.device) -> int:
    """
    Generates a prediction for a single multiple-choice example.

    Args:
        model: The HuggingFace multiple-choice model.
        tokenizer: The tokenizer associated with the model.
        sentence: The input sentence containing a blank (to be split).
        option1: The first candidate choice.
        option2: The second candidate choice.
        max_length: Maximum token length for the tokenizer.
        device: The device (CPU/GPU/MPS) to run the model on.

    Returns:
        int: The index of the predicted option (0 for option1, 1 for option2).
    """
    # Split the sentence at the blank character (_)
    before, after = split_sentence(sentence)

    # Format the inputs for the MultipleChoice model
    # The model expects paired inputs: (context, continuation)
    first_sentences = [before, before]
    second_sentences = [f"{option1} {after}".strip(), f"{option2} {after}".strip()]

    # Tokenize the pairs
    enc = tokenizer(
        first_sentences,
        second_sentences,
        truncation=True,
        max_length=max_length,
        padding=True,
        return_tensors="pt",
    )

    # MultipleChoice models expect an extra dimension for the number of choices.
    # We unsqueeze(0) to add the batch dimension: shape becomes (1, num_choices, seq_length)
    enc = {k: v.unsqueeze(0).to(device) for k, v in enc.items()}

    # Perform inference without computing gradients
    with torch.no_grad():
        logits = model(**enc).logits

    # Return the index of the choice with the highest logit score
    return int(torch.argmax(logits, dim=1).item())


def has_gold_answer(answer) -> bool:
    """
    Checks if a valid gold answer is provided in the dataset.
    This is useful for test sets where ground truth labels might be hidden or missing.

    Args:
        answer: The answer value from the dataset.

    Returns:
        bool: True if the answer is explicitly "1" or "2", False otherwise.
    """
    if answer is None:
        return False
    answer_str = str(answer).strip()
    return answer_str in {"1", "2"}


def main() -> None:
    """
    Main evaluation pipeline:
    1. Sets up the device and loads data.
    2. Loads the fine-tuned model and tokenizer.
    3. Iterates through the dataset to generate predictions.
    4. Computes accuracy (if labels are available).
    5. Exports predictions and performance metrics to disk.
    """
    args = parse_args()

    # Determine the best available hardware accelerator
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    # Load dataset and select the appropriate split
    ds = load_local_winogrande(args.data_dir)
    split_ds = ds[args.split]

    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForMultipleChoice.from_pretrained(args.model_dir).to(device)
    model.eval()

    rows = []
    submission_rows = []
    correct = 0
    n_with_gold = 0

    # Inference loop
    for idx, ex in enumerate(tqdm(split_ds, desc=f"Evaluation {args.split}")):
        # Get model prediction (0 or 1)
        pred = predict_one(
            model=model,
            tokenizer=tokenizer,
            sentence=ex["sentence"],
            option1=ex["option1"],
            option2=ex["option2"],
            max_length=args.max_length,
            device=device,
        )
        
        # Convert prediction back to string format ("1" or "2") for submission
        pred_answer = str(pred + 1)
        submission_rows.append({"idx": idx, "answer": pred_answer})

        gold = None
        is_ok = None
        
        # Check against ground truth if available
        if has_gold_answer(ex.get("answer")):
            gold = answer_to_label(ex["answer"])  # Converts "1"/"2" to 0/1
            is_ok = int(pred == gold)
            correct += is_ok
            n_with_gold += 1

        # Store detailed results for analysis
        rows.append(
            {
                "idx": idx,
                "sentence": ex["sentence"],
                "option1": ex["option1"],
                "option2": ex["option2"],
                "answer": ex["answer"],
                "gold_label": gold,
                "pred_label": pred,
                "pred_answer": pred_answer,
                "correct": is_ok,
            }
        )

    # Compute overall accuracy
    accuracy = (correct / n_with_gold) if n_with_gold > 0 else None

    # Ensure output directories exist
    pred_dir = ensure_dir(args.pred_dir)
    report_dir = ensure_dir(args.report_dir)

    # Define output file paths
    pred_path = Path(pred_dir) / f"pred_{args.split}.csv"
    submission_path = Path(pred_dir) / f"submission_{args.split}.csv"
    report_path = Path(report_dir) / f"eval_{args.split}.json"

    # Export detailed predictions and formatted submission files
    pd.DataFrame(rows).to_csv(pred_path, index=False)
    pd.DataFrame(submission_rows).to_csv(submission_path, index=False)
    
    # Export evaluation metrics
    save_json({"split": args.split, "accuracy": accuracy, "n": len(rows), "n_with_gold": n_with_gold}, report_path)

    # Print summary to console
    if accuracy is None:
        print(f"Accuracy {args.split}: n/a (pas de labels gold)")
    else:
        print(f"Accuracy {args.split}: {accuracy:.4f}")
    print(f"Predictions: {pred_path}")
    print(f"Submission: {submission_path}")
    print(f"Rapport: {report_path}")


if __name__ == "__main__":
    main()