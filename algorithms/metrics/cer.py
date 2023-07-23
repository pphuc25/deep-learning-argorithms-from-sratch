import numpy as np

def cer_metric(predict_sentence: str, reference_sentence: str) -> float:
    """
    Compute the character error rate (WER) between predict sentence and
    reference sentence

    Args:
        predict_sentence: The predict sentence
        reference_sentence: The reference sentence

    Returns:
        (float) The wer score of predict sentence according to reference sentence
    """
    len_refer, len_pred = len(reference_sentence), len(predict_sentence)
    
    matrix_subtask = np.zeros((len_refer+1, len_pred+1))
    for i in range(len_refer + 1): matrix_subtask[i, 0] = i
    for j in range(len_pred + 1): matrix_subtask[0, j] = j

    for i in range(1, len_refer + 1):
        for j in range(1, len_pred + 1):
            if reference_sentence[i-1] == predict_sentence[j-1]:
                matrix_subtask[i][j] = matrix_subtask[i-1][j-1]
            else:
                substitutions = matrix_subtask[i-1][j-1] + 1
                deletions = matrix_subtask[i][j-1] + 1
                insertions = matrix_subtask[i-1][j] + 1
                matrix_subtask[i][j] = min(substitutions, deletions, insertions)
    cer_score = matrix_subtask[-1][-1] / len_refer
    return cer_score