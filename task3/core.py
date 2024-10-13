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
    prediction = re.sub(r'\W+', ' ', prediction).lower().strip()
    ground_truth = re.sub(r'\W+', ' ', ground_truth).lower().strip()

    gt_tokens = ground_truth.split()
    gt_n_grams = [' '.join(gt_tokens[i:i + N]) for i in range(len(gt_tokens) - N + 1)]

    pred_tokens = prediction.split()
    pred_n_grams = [' '.join(pred_tokens[i:i + N]) for i in range(len(pred_tokens) - N + 1)]

    # recall
    overlap = len([ngram for ngram in pred_n_grams if ngram in gt_n_grams])
    recall = overlap / len(gt_n_grams) if len(gt_n_grams) > 0 else 0

    # precision
    precision = overlap / len(pred_n_grams) if len(pred_n_grams) > 0 else 0

    # ROUGE-N score
    rouge = 0 if abs(recall + precision) <= 1e-16 else 2 * (precision * recall) / (precision + recall)
    ################################################

    return rouge