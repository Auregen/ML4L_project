from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import yaml


def ensure_dir(path: str | Path) -> Path:
    """
    Ensures that a directory exists, creating it and its parent directories if necessary.
    
    Args:
        path (str | Path): The target directory path.
        
    Returns:
        Path: A Path object representing the target directory.
    """
    out = Path(path)
    # Create the directory structure. 
    # parents=True creates missing intermediate directories.
    # exist_ok=True prevents raising a FileExistsError if the target already exists.
    out.mkdir(parents=True, exist_ok=True)
    return out


def load_yaml(path: str | Path) -> Dict[str, Any]:
    """
    Reads a YAML file and parses its contents into a Python dictionary.
    
    Args:
        path (str | Path): The path to the YAML file to read.
        
    Returns:
        Dict[str, Any]: A dictionary containing the parsed YAML data.
    """
    with Path(path).open("r", encoding="utf-8") as f:
        # safe_load is used instead of load to prevent the execution of 
        # arbitrary Python objects embedded within the YAML file.
        return yaml.safe_load(f)


def save_json(obj: Dict[str, Any], path: str | Path) -> None:
    """
    Serializes a Python dictionary and saves it as a formatted JSON file.
    
    Args:
        obj (Dict[str, Any]): The data dictionary to serialize.
        path (str | Path): The destination path for the JSON file.
    """
    with Path(path).open("w", encoding="utf-8") as f:
        # indent=2 adds line breaks and indentation for human readability.
        # ensure_ascii=False writes non-ASCII characters (like accents) as-is 
        # rather than escaping them (e.g., as \uXXXX).
        json.dump(obj, f, indent=2, ensure_ascii=False)