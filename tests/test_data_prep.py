from src.data_prep import answer_to_label, split_sentence


def test_split_sentence_with_blank():
    """
    Tests the `split_sentence` function when the input string contains an 
    underscore ('_') representing a fill-in-the-blank space.
    """
    # Call the function with a sentence containing a blank.
    before, after = split_sentence("John gave _ a gift")
    
    # Verify that the text preceding the blank is correctly extracted.
    assert before == "John gave"
    
    # Verify that the text following the blank is correctly extracted.
    assert after == "a gift"


def test_split_sentence_without_blank():
    """
    Tests the `split_sentence` function when the input string does not 
    contain any underscore/blank character.
    """
    # Call the function with a standard sentence.
    before, after = split_sentence("No blank here")
    
    # Verify that the entire string is assigned to the 'before' variable.
    assert before == "No blank here"
    
    # Verify that the 'after' variable is simply an empty string.
    assert after == ""


def test_answer_to_label_ok():
    """
    Tests the `answer_to_label` function to ensure it correctly maps 
    1-based string answers to 0-based integer labels.
    """
    # Verify that the string "1" is converted to the integer label 0.
    assert answer_to_label("1") == 0
    
    # Verify that the string "2" is converted to the integer label 1.
    assert answer_to_label("2") == 1