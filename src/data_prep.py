from __future__ import annotations

from typing import Dict, List, Tuple

from datasets import Dataset, DatasetDict, load_dataset

# Columns required to process the Winogrande dataset
REQUIRED_COLUMNS = ["sentence", "option1", "option2", "answer"]


def split_sentence(sentence: str) -> Tuple[str, str]:
    """
    Splits a fill-in-the-blank sentence around the underscore character ('_').
    
    Args:
        sentence: The input string containing an underscore representing the missing word.
        
    Returns:
        A tuple containing the text before the underscore and the text after it.
    """
    parts = sentence.split("_")
    before = parts[0].rstrip()
    after = parts[1].lstrip() if len(parts) > 1 else ""
    return before, after


def answer_to_label(answer) -> int:
    """
    Converts a string-based answer into a zero-indexed integer label for model training.
    
    Args:
        answer: The expected answer, typically "1" or "2".
        
    Returns:
        0 if the answer is "1", and 1 if the answer is "2".
        
    Raises:
        ValueError: If the answer is neither "1" nor "2".
    """
    answer_str = str(answer).strip()
    if answer_str == "1":
        return 0
    if answer_str == "2":
        return 1
    raise ValueError(f"Unexpected answer value: {answer}")


def load_local_winogrande(data_dir: str) -> DatasetDict:
    """
    Loads the Winogrande dataset splits (train, validation, test) from local CSV files.
    
    Args:
        data_dir: The directory path containing the CSV files.
        
    Returns:
        A Hugging Face DatasetDict containing the loaded data.
        
    Raises:
        ValueError: If any of the required columns are missing in any of the splits.
    """
    data_files = {
        "train": f"{data_dir}/winogrande_train.csv",
        "validation": f"{data_dir}/winogrande_validation.csv",
        "test": f"{data_dir}/winogrande_test.csv",
    }
    ds = load_dataset("csv", data_files=data_files)
    
    # Validate the structure of each split
    for split in ["train", "validation", "test"]:
        for col in REQUIRED_COLUMNS:
            if col not in ds[split].column_names:
                raise ValueError(f"Missing column in {split} split: {col}")
    return ds


def build_gold_labels(dataset: Dataset) -> List[int]:
    """
    Extracts and converts all answers from a dataset into a list of integer labels.
    
    Args:
        dataset: A single Hugging Face Dataset split.
        
    Returns:
        A list of integer labels (0 or 1).
    """
    return [answer_to_label(a) for a in dataset["answer"]]


def preprocess_for_multiple_choice(batch: Dict[str, List[str]], tokenizer, max_length: int) -> Dict[str, List[List[int]]]:
    """
    Preprocesses a batch of examples into the format required for multiple-choice models.
    It unrolls the two options to pair them with the context, tokenizes them, and regroups
    them into a [batch_size, num_choices, sequence_length] format.
    
    Args:
        batch: A dictionary containing lists of "sentence", "option1", "option2", and "answer".
        tokenizer: The Hugging Face tokenizer to use.
        max_length: The maximum sequence length for truncation.
        
    Returns:
        A dictionary of tokenized features, including input_ids, attention_mask, and labels.
    """
    first_sentences: List[str] = []
    second_sentences: List[str] = []

    # Flatten the inputs: each example generates 2 pairs of (context, option)
    for sentence, option1, option2 in zip(batch["sentence"], batch["option1"], batch["option2"]):
        before, after = split_sentence(sentence)

        # The context remains the same for both options
        first_sentences.extend([before, before])
        
        # The option completes the sentence alongside the remaining context
        second_sentences.extend([
            f"{option1} {after}".strip(),
            f"{option2} {after}".strip(),
        ])

    # Tokenize the flattened pairs
    tokenized = tokenizer(
        first_sentences,
        second_sentences,
        truncation=True,
        max_length=max_length,
    )

    num_examples = len(batch["sentence"])
    
    # Regroup the 1D lists back into 2D lists of shape (batch_size, 2, sequence_length)
    features = {
        key: [tokenized[key][i : i + 2] for i in range(0, 2 * num_examples, 2)]
        for key in tokenized
    }
    
    # Append the integer labels to the final feature dictionary
    features["label"] = [answer_to_label(a) for a in batch["answer"]]
    return features