from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from scipy.stats import binomtest, chi2


def mcnemar_table(correct_a: pd.Series, correct_b: pd.Series) -> tuple[int, int, int, int]:
    """
    Computes the 2x2 contingency table counts for McNemar's test.
    
    Args:
        correct_a: Boolean or binary series representing correct predictions for model A.
        correct_b: Boolean or binary series representing correct predictions for model B.
        
    Returns:
        A tuple containing counts for (both_correct, a_only, b_only, both_wrong).
    """
    # Count instances where both models predicted correctly
    both_correct = int(((correct_a == 1) & (correct_b == 1)).sum())
    
    # Count instances where only model A was correct (first discordant pair)
    a_only = int(((correct_a == 1) & (correct_b == 0)).sum())
    
    # Count instances where only model B was correct (second discordant pair)
    b_only = int(((correct_a == 0) & (correct_b == 1)).sum())
    
    # Count instances where both models predicted incorrectly
    both_wrong = int(((correct_a == 0) & (correct_b == 0)).sum())
    
    return both_correct, a_only, b_only, both_wrong


def mcnemar_pvalues(b: int, c: int) -> tuple[float, float]:
    """
    Calculates the p-values for McNemar's test using the discordant counts.
    
    Args:
        b: Count of instances where only model A is correct (a_only).
        c: Count of instances where only model B is correct (b_only).
        
    Returns:
        A tuple containing the Chi-squared p-value (with continuity correction) 
        and the exact binomial p-value.
    """
    # If there are no discordant pairs, the models have identical error profiles
    if b + c == 0:
        return 1.0, 1.0

    # Calculate Chi-squared statistic with Edwards' continuity correction
    # Formula: (|b - c| - 1)^2 / (b + c)
    chi2_stat = ((abs(b - c) - 1) ** 2) / (b + c)
    
    # Compute the survival function (1 - CDF) to get the p-value for 1 degree of freedom
    p_chi2 = float(chi2.sf(chi2_stat, df=1))

    # Calculate the exact p-value using a two-sided binomial test.
    # This evaluates if the probability of success for discordant pairs differs from 0.5.
    exact = binomtest(k=min(b, c), n=b + c, p=0.5, alternative="two-sided")
    p_exact = float(exact.pvalue)

    return p_chi2, p_exact


def parse_args() -> argparse.Namespace:
    """
    Parses command-line arguments for the script.
    """
    parser = argparse.ArgumentParser(description="McNemar test between two prediction files")
    
    # Required arguments for the input CSV files containing model predictions
    parser.add_argument("--pred-a", type=str, required=True, help="CSV predictions for model A")
    parser.add_argument("--pred-b", type=str, required=True, help="CSV predictions for model B")
    
    # Optional argument to specify the output report path
    parser.add_argument("--out", type=str, default="outputs/reports/mcnemar_report.txt", help="Output file path")
    
    return parser.parse_args()


def main() -> None:
    """
    Main execution pipeline: loads data, validates columns, computes McNemar 
    statistics, and exports a formatted text report.
    """
    args = parse_args()

    # Load prediction dataframes from the provided CSV paths
    df_a = pd.read_csv(args.pred_a)
    df_b = pd.read_csv(args.pred_b)

    # Validate that both datasets have the exact same number of observations
    if len(df_a) != len(df_b):
        raise ValueError("The two files do not have the same number of rows")

    # Ensure the required columns ('idx' for merging, 'correct' for evaluation) exist
    required_cols = {"idx", "correct"}
    if not required_cols.issubset(set(df_a.columns)):
        raise ValueError("pred-a must contain 'idx' and 'correct' columns")
    if not required_cols.issubset(set(df_b.columns)):
        raise ValueError("pred-b must contain 'idx' and 'correct' columns")

    # Merge the two dataframes on the index column to align the predictions row by row
    merged = df_a[["idx", "correct"]].merge(df_b[["idx", "correct"]], on="idx", suffixes=("_a", "_b"))

    # Compute the contingency table counts and the statistical p-values
    both_correct, a_only, b_only, both_wrong = mcnemar_table(merged["correct_a"], merged["correct_b"])
    p_chi2, p_exact = mcnemar_pvalues(a_only, b_only)

    # Prepare the formatted lines for the output report
    report_lines = [
        "McNemar report",
        f"n={len(merged)}",
        f"both_correct={both_correct}",
        f"a_only={a_only}",
        f"b_only={b_only}",
        f"both_wrong={both_wrong}",
        f"p_chi2_cc={p_chi2:.6g}",  # Chi-squared p-value with continuity correction
        f"p_exact={p_exact:.6g}",   # Exact binomial p-value
    ]

    # Create the output directory tree if it doesn't exist, then write the report
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    # Print the report to the console and confirm the save location
    print("\n".join(report_lines))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()