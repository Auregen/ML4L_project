from stats.mcnemar import mcnemar_pvalues, mcnemar_table
import pandas as pd

def test_mcnemar_table_counts():
    """
    Tests the `mcnemar_table` function to ensure it correctly computes 
    the 2x2 contingency table counts for two sets of binary outcomes.
    """
    # Define mock binary predictions for two models (A and B).
    # Typically, 1 represents a correct prediction and 0 an incorrect one.
    a = pd.Series([1, 1, 0, 0, 1])
    b = pd.Series([1, 0, 1, 0, 1])
    
    # Calculate the contingency table values comparing model A and model B.
    both_correct, a_only, b_only, both_wrong = mcnemar_table(a, b)
    
    # Assertions to verify the output matches the expected manual counts:
    
    # 2 instances where both A and B are 1 (indices 0 and 4)
    assert both_correct == 2
    
    # 1 instance where A is 1 and B is 0 (index 1)
    assert a_only == 1
    
    # 1 instance where A is 0 and B is 1 (index 2)
    assert b_only == 1
    
    # 1 instance where both A and B are 0 (index 3)
    assert both_wrong == 1


def test_mcnemar_pvalues_in_range():
    """
    Tests the `mcnemar_pvalues` function to guarantee that the returned 
    p-values are mathematically valid probabilities (between 0.0 and 1.0).
    """
    # Compute p-values using arbitrary discordant cell counts (e.g., 10 and 5).
    # - p_chi2 corresponds to the asymptotic approximation (Chi-squared).
    # - p_exact corresponds to the exact binomial test.
    p_chi2, p_exact = mcnemar_pvalues(10, 5)
    
    # Validate that the chi-squared p-value is within the [0, 1] range.
    assert 0.0 <= p_chi2 <= 1.0
    
    # Validate that the exact binomial p-value is within the [0, 1] range.
    assert 0.0 <= p_exact <= 1.0