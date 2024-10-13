import re

def calculate_rouge_n(prediction, ground_truth, N):
    """
    Calculate the ROUGE-N score for the given prediction and ground truth strings.

    Args:
    - prediction (str): The predicted text.
    - ground_truth (str): The reference or ground truth text.
    - N (int): The number of grams to consider (e.g., 1 for ROUGE-1, 2 for ROUGE-2).

    Returns:
    - float: The ROUGE-N score.
    """
    gt_n_grams = []
    pred_n_grams = []
    recall = 0
    precision = 0
    rouge = None

    ################################################
    # TODO
    # Filter out non-alphanumeric characters from the prediction and ground_truth strings, leaving only word characters (letters, digits, and underscores) and whitespace
    prediction = re.sub(r'\W+', ' ', prediction).lower().strip()
    ground_truth = re.sub(r'\W+', ' ', ground_truth).lower().strip()

    # Create n-grams for ground truth
    gt_tokens = ground_truth.split()
    gt_n_grams = [' '.join(gt_tokens[i:i + N]) for i in range(len(gt_tokens) - N + 1)]

    # Create n-grams for prediction
    pred_tokens = prediction.split()
    pred_n_grams = [' '.join(pred_tokens[i:i + N]) for i in range(len(pred_tokens) - N + 1)]

    # Calculate recall
    overlap = len([ngram for ngram in pred_n_grams if ngram in gt_n_grams])
    recall = overlap / len(gt_n_grams) if len(gt_n_grams) > 0 else 0.0

    # Calculate precision
    precision = overlap / len(pred_n_grams) if len(pred_n_grams) > 0 else 0.0

    # Calculate ROUGE-N score
    if recall + precision == 0:
        rouge = 0.0
    else:
        rouge = 2 * (precision * recall) / (precision + recall)
    ################################################

    return rouge