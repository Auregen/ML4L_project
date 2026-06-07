from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from transformers import (
    AutoModelForMultipleChoice,
    AutoTokenizer,
    DataCollatorForMultipleChoice,
    Trainer,
    TrainingArguments,
)

from src.data_prep import load_local_winogrande, preprocess_for_multiple_choice
from src.io_utils import ensure_dir, load_yaml, save_json


def set_seed(seed: int) -> None:
    """
    Sets the seed for random number generators to ensure reproducibility.
    Applies the seed to standard Python random, NumPy, and PyTorch (CPU and GPU).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    """
    Parses command-line arguments required to run the training script.
    
    Returns:
        argparse.Namespace: Object containing the parsed arguments (config path, data dirs, etc.).
    """
    parser = argparse.ArgumentParser(description="Multiple-Choice Fine-Tuning for WinoGrande")
    
    # Required argument: path to the YAML configuration file containing hyperparameters
    parser.add_argument("--config", type=str, required=True, help="Path to the experiment YAML config")
    
    # Optional arguments with default paths for data, models, and reports
    parser.add_argument("--data-dir", type=str, default="data/raw", help="Directory containing raw CSV data")
    parser.add_argument("--output-dir", type=str, default="outputs/models/run_mcq", help="Directory to save the trained model")
    parser.add_argument("--report-dir", type=str, default="outputs/reports", help="Directory to save evaluation reports")
    
    return parser.parse_args()


def main() -> None:
    """
    Main execution function: loads data, initializes the model, runs the training loop,
    evaluates the model, and saves the artifacts.
    """
    # Parse CLI arguments and load hyperparameters from the YAML config file
    args = parse_args()
    cfg = load_yaml(args.config)

    # Set random seed for reproducible training runs
    set_seed(int(cfg["seed"]))
    
    # Load the WinoGrande dataset from local files
    ds = load_local_winogrande(args.data_dir)

    # Initialize the tokenizer and model from Hugging Face using the specified architecture
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    model = AutoModelForMultipleChoice.from_pretrained(cfg["model_name"])

    # Define a lambda function to pass the tokenizer and max_length to the preprocessing function
    preprocess = lambda b: preprocess_for_multiple_choice(b, tokenizer, int(cfg["max_length"]))
    
    # Apply tokenization and formatting to both train and validation splits
    # 'remove_columns' drops the original raw text columns to keep only tensors (input_ids, attention_mask, etc.)
    tokenized_train = ds["train"].map(preprocess, batched=True, remove_columns=ds["train"].column_names)
    tokenized_val = ds["validation"].map(preprocess, batched=True, remove_columns=ds["validation"].column_names)

    # Initialize the data collator specific to multiple-choice tasks 
    # (handles dynamic padding of multiple choices per batch)
    data_collator = DataCollatorForMultipleChoice(tokenizer=tokenizer)
    
    def compute_metrics(eval_pred):
        """
        Calculates custom metrics (Accuracy) during the evaluation phase.
        Compares model predictions (highest logit) against true labels.
        """
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)
        return {"accuracy": float((preds == labels).mean())}

    # Ensure output directories exist before attempting to save files there
    out_model_dir = ensure_dir(args.output_dir)
    out_report_dir = ensure_dir(args.report_dir)

    # Configure the arguments for the Hugging Face Trainer using parsed YAML values
    training_args = TrainingArguments(
        output_dir=str(out_model_dir),
        learning_rate=float(cfg["learning_rate"]),
        weight_decay=float(cfg["weight_decay"]),
        num_train_epochs=int(cfg["num_train_epochs"]),
        per_device_train_batch_size=int(cfg["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(cfg["per_device_eval_batch_size"]),
        gradient_accumulation_steps=int(cfg["gradient_accumulation_steps"]),
        warmup_ratio=float(cfg["warmup_ratio"]),
        fp16=bool(cfg["fp16"]),  # Enable mixed precision training if True
        eval_strategy=str(cfg["eval_strategy"]),
        save_strategy=str(cfg["save_strategy"]),
        metric_for_best_model=str(cfg["metric_for_best_model"]),
        greater_is_better=bool(cfg["greater_is_better"]),
        load_best_model_at_end=bool(cfg["load_best_model_at_end"]),
        logging_steps=int(cfg["logging_steps"]),
        report_to="none",  # Disables logging to W&B, TensorBoard, etc.
        seed=int(cfg["seed"]),
    )

    # Instantiate the Trainer with the model, datasets, tokenizer, and metrics
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Execute the training loop
    train_result = trainer.train()
    
    # Run a final evaluation on the validation dataset
    eval_result = trainer.evaluate(tokenized_val)

    # Save the final trained model and tokenizer to the designated output directory
    trainer.save_model(str(out_model_dir))
    tokenizer.save_pretrained(str(out_model_dir))

    # Compile training and evaluation metrics into a single dictionary
    # Filters out non-numeric values from eval_result to ensure JSON serialization
    save_json({
        "train_loss": float(train_result.training_loss),
        **{k: float(v) for k, v in eval_result.items() if isinstance(v, (int, float))},
    }, Path(out_report_dir) / "train_eval_summary.json")

    # Output final confirmation paths
    print("Training finished.")
    print(f"Model saved to: {out_model_dir}")
    print(f"Report saved to: {Path(out_report_dir) / 'train_eval_summary.json'}")


if __name__ == "__main__":
    main()