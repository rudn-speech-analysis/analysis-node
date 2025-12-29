import re
import numpy as np

def normalize_text(text):
    """
    Normalize the text for WER calculation:
    - Convert to lowercase
    - Remove punctuation and extra whitespace
    - Split into words
    """
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    words = text.strip().split()
    return words

def compute_wer(reference, hypothesis):
    """
    Compute the Word Error Rate (WER) between reference and hypothesis texts.
    
    Args:
    reference (str): The ground truth transcription text.
    hypothesis (str): The Whisper-generated transcription text.
    
    Returns:
    float: The WER value (between 0.0 and 1.0, or potentially higher if many insertions).
    
    This function uses dynamic programming to compute the Levenshtein distance
    at the word level, which gives the number of substitutions, insertions, and
    deletions. WER is then (S + D + I) / N, where N is the number of words in
    the reference.
    
    Texts are normalized by lowercasing and removing punctuation before comparison.
    """
    ref_words = normalize_text(reference)
    hyp_words = normalize_text(hypothesis)
    
    # If reference is empty, WER is undefined, but we can return infinity or a large number
    if len(ref_words) == 0:
        return float('inf') if len(hyp_words) > 0 else 0.0
    
    # Create a DP table: dp[i][j] will be the edit distance between first i ref words and first j hyp words
    m, n = len(ref_words) + 1, len(hyp_words) + 1
    dp = np.zeros((m, n), dtype=int)
    
    # Initialize base cases
    for i in range(m):
        dp[i][0] = i  # Deletions
    for j in range(n):
        dp[0][j] = j  # Insertions
    
    # Fill the DP table
    for i in range(1, m):
        for j in range(1, n):
            if ref_words[i-1] == hyp_words[j-1]:
                cost = 0  # Match
            else:
                cost = 1  # Substitution
            dp[i][j] = min(
                dp[i-1][j] + 1,    # Deletion
                dp[i][j-1] + 1,    # Insertion
                dp[i-1][j-1] + cost  # Substitution or match
            )
    
    # The total errors are the edit distance
    errors = dp[m-1][n-1]
    
    # WER = errors / number of reference words
    wer = errors / len(ref_words)
    return wer

# Example usage (for testing):
# ref = "this is a test"
# hyp = "this is test"
# print(compute_wer(ref, hyp))  # Should be around 0.25 (1 deletion / 4 words)